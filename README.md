# radocorrect

Radocorrect is a tool which corrects radolan precipitation estimates from DWD using the ground-truth measurements at the rain gauges. The **aim** is to retrieve hourly gridded precipitation information which e.g. can be applied in hydrological models.

## Approach
1.  the algorithm computes a weighting factor for each time step at the rain gauge locations by comparing radolan and gauge estimates.
2. The weighting factors are distributed spatially on the radolan grid using inverse distance weighting
3. At each radolan grid cell, the radolan precipitation estimate is multiplied with the corresponding weighting factor
4. Spatio-temporal datasets are stacked as xarray dataset which can be exported to netcdf files

##Dependencies
* roverweb 
* nrt_io
* for others, see requirements.txt

## Limitations
* IDW Interpolation only
* No in-depth analysis of deviation between radolan and gauge depending on rain pattern structure (e.g duration, intensity of event)
* Outlier (very high weighting factors) can remove by setting percentile threshold, only
* Domain boundarys have to be provided as shapefiles

## Code Example
```python
import radocorrect
radocorrect.radocorrect(start_time='2007-11-05:1500',
              end_time='2007-11-06:1800',
              domain_path='.\\Input\\Mueglitz_basin_grid.shp',
              dwd_search_area_path='.\\Input\\dwd_rectangle.shp',
              no_of_nearest_stations=5,
              thresh_percentile=0.95,
              output=True)
```
## Authors

* **Erik Nixdorf**

## Acknowledgments

* Thx to all who helped to improve the code


