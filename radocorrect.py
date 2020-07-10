"""
This tool aims to correct the radolan data with the data from the DWD Stations
10 min time shift has to be accepted as error
author: Erik Nixdorf
Version: 0.1
Legal rights belong to: UFZ & Erik Nixdorf

"""

from nrt_io import downloader
from pyproj import Proj, Transformer,CRS
import numpy as np
import geopandas as gpd
from roverweb import weather
from datetime import datetime
import os
import pandas as pd
import xarray as xr
from scipy import interpolate
def get_nearest_indices(row,geometry_column=['geometry_radolan_x'],coordinate_array=[np.zeros((5,5))]):
    """
    A Tool which finds closest column/row indices in a matrix by minimum distance
    Parameters based on data from rows in a dataframe
    ----------
    row : TYPE
        DESCRIPTION.
    geometry_column : TYPE, optional
        DESCRIPTION. The default is ['geometry_radolan_x'].
    coordinate_array : TYPE, optional
        DESCRIPTION. The default is [rado_y].
    Returns
    -------
    index : TYPE
        DESCRIPTION.
    """
    index=list()
    for i in range(0,len(geometry_column)):
        index.append(np.argmin(np.abs(coordinate_array[i]-row[geometry_column[i]])))
    return index


def Interpolation_2D(dataarray,method='cubic'):
    """  
    A tool to interpolate over the nans for each time step
    Parameters
    inspired by https://modelhelptokyo.wordpress.com/2017/10/25/how-to-interpolate-missing-values-2d-python/
    ----------
    dataarray : xr dataarray
        A 2D Dataarray containing nans
    method : string
        DESCRIPTION. Method for grid interpolation using 

    Returns
    -------
    interpolated_array : np.array
        The np. array where nans are replaced

    """
    x = dataarray.coords['x'].values
    y = dataarray.coords['y'].values
    #mask invalid values
    array=np.array(dataarray)
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    
    interpolated_array = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method=method)
    return interpolated_array    
#%% setting
start_time='2007-01-31:1500'
end_time='2007-02-03:0300'
domain_path='.\\Input\\Mueglitz_basin_grid.shp'
dwd_search_area_path='.\\Input\\dwd_rectangle.shp'
no_of_nearest_stations=5

#%% Get the radolan data for a certain time period

rado_init=downloader.downloader(start_time=start_time, end_time=end_time,roi=dwd_search_area_path,roi_buffer=0.01)

rado_rain_raw=rado_init.radorecent(time_res='hourly',to_harddisk=True)

rado_rain=10*rado_rain_raw.RW # to mm
# make invalid values to nan
rado_rain=rado_rain.where(rado_rain>=0,drop=True)

#manipulate the time by 10 min
rado_rain.coords['time'].values=rado_rain.coords['time'].values+np.timedelta64(10,'m')
#get rado coordinates
rado_x=rado_rain.coords['x'].values
rado_y=rado_rain.coords['y'].values
#create empty array which will be for the factor between station and radolan data
rain_multiplicator = xr.DataArray(np.empty([rado_rain.sizes['time'],rado_rain.sizes['y'],rado_rain.sizes['x']])*np.nan,
                    dims=rado_rain.dims,
                    coords=rado_rain.coords)



#%%get the data from dwd stations
#import the grid
domain=gpd.GeoDataFrame.from_file(domain_path)
#find nearest stations
test,dwd_base=weather.Find_nearest_dwd_stations(domain,
    date_start=datetime.strptime(start_time,'%Y-%m-%d:%H%M').date().strftime('%Y%m%d'),
    date_end=datetime.strptime(end_time,'%Y-%m-%d:%H%M').date().strftime('%Y%m%d'),
    dwd_time_format='%Y%m%d%H',
    data_category='precipitation',
    temp_resolution='hourly',
    no_of_nearest_stations=no_of_nearest_stations,
    memory_save=True,
    Output='True')

#cut the dwd database to the actual dates of the radolan dataset
dwd_base=dwd_base.sel(time=rado_rain.coords['time'].values)
print(str(len(dwd_base.STATIONS_ID)),'Stations with valid rainfall data used')
#%%load the station network  and check their positions on radogrid
#check file
files=os.listdir('.\\roverweb\\tables\\')
dwd_station_file= [file for file in files if '.csv' in file][0]
dwd_stations=pd.read_csv('.\\roverweb\\tables\\'+dwd_station_file)
#select rows based on dwd data
rel_station_list=list(np.array(dwd_base.STATIONS_ID))
dwd_stations=dwd_stations.loc[dwd_stations['STATIONS_ID'].isin(rel_station_list)]
dwd_stations['geo_x']=np.nan
dwd_stations['geo_y']=np.nan
#transform the coordinates to radolan
#define the radolan_projection
rado_proj_string='+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 + x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs'
rado_proj = CRS(rado_proj_string)
#define a transformer to deal with the differenct CRS
transformer = Transformer.from_crs(CRS('epsg:4326'), rado_proj, always_xy=True)
#convert the bounds str to radolan projection
dwd_lonlat=dwd_stations[['geo_lon','geo_lat']]
dwd_stations[['geo_x','geo_y']]=np.array([transformer.transform(*xy) for xy in np.array(dwd_stations[['geo_lon','geo_lat']])]) 

#%%find the columns and rows in radolan

position_indices=np.array(dwd_stations.apply(get_nearest_indices,axis=1,geometry_column=['geo_x','geo_y'],coordinate_array=[rado_x,rado_y]).to_list())
# add station ID as last column
position_indices=np.append(position_indices,np.array(dwd_stations.STATIONS_ID).reshape(np.size(position_indices,0),1),axis=1)
#next we loop with isel through the entire rado_rain to get hour data
for station in position_indices:
    station_rado=rado_rain.isel(x=station[0],y=station[1])
    #replace zero with 0.01 to avoid 0/0 division
    station_rado=station_rado.where(station_rado!=0,other=0.01)
    #same for dwd data
    station_measured=dwd_base.sel(STATIONS_ID=station[2]).hourly_precipitation_height
    station_measured=station_measured.where(station_measured!=0,other=0.01)
    #get the multiplicator for each time step
    station_multiplicator=station_measured/station_rado
    #replace the entry in the multiplicator dataarray, remember that variable order is time, y, x
    rain_multiplicator[:,station[1],station[0]]=station_multiplicator
    

#%% Next we interpolate over the nans using a loop, apply is possible as well
for time_step in range(0,rain_multiplicator.sizes['time']):
    #check whether at least for the minimum number of stations exist non-nan values
    #because the problem is that radolan has invalid data nan
    if np.count_nonzero(~np.isnan(rain_multiplicator[time_step,:,:]))<no_of_nearest_stations:
        print('Not enough valid points to interpolated for time step ',str(time_step))
    else:   
        interpolated_grid=Interpolation_2D(rain_multiplicator[time_step,:,:],method='linear')
        # we have to interpolate a second time to extrapolate all remaining nans, which are outside convex hull
        # discussion @ https://stackoverflow.com/questions/21993655/interpolation-and-extrapolation-of-randomly-scattered-data-to-uniform-grid-in-3d
        nearest_grid=Interpolation_2D(rain_multiplicator[time_step,:,:],method='nearest')
        #replace the nan of initial interpolation with the ones from nearest grid
        nan_locations=np.isnan(interpolated_grid)
        interpolated_grid[nan_locations]=nearest_grid[nan_locations]
        rain_multiplicator[time_step,:,:]=interpolated_grid
    
#%% reduce data array dimension to actual roi coordinates
#get domain bounds
domain_loader=downloader.downloader(start_time=start_time, end_time=end_time,roi=domain_path,roi_buffer=0.01)
domain_bounds_rado=np.array([transformer.transform(*xy) for xy in domain_loader.roi_bounds])

rain_multiplicator=rain_multiplicator.sel(x=slice(domain_bounds_rado[0,0],domain_bounds_rado[1,0]),
                                          y=slice(domain_bounds_rado[0,1],domain_bounds_rado[1,1])
                                          )
rado_rain=rado_rain.sel(x=slice(domain_bounds_rado[0,0],domain_bounds_rado[1,0]),
                                          y=slice(domain_bounds_rado[0,1],domain_bounds_rado[1,1])
                                          )

#%% multiply datasets and append to original xarray
rado_corrected=rado_rain[:,:,:]*rain_multiplicator[:,:,:]
#%% write final dataset
ds_out = rado_rain.to_dataset(name = 'radolan')
ds_out['correction_factor']=rain_multiplicator
ds_out['radolan_corrected']=rado_corrected
#sort the time
ds_out=ds_out.sortby('time')
#write it out
os.makedirs("Output", exist_ok=True) 
ds_out.to_netcdf('.\\Output\\radolan_correct_'+start_time[0:10]+'_'+end_time[0:10]+'.nc')


