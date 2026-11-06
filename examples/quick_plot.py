

#Import general libs you'll need
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# Import functions from main
from phoreal.agg_configs import ATL_AGG_CONFIG, ALS_AGG_CONFIG, ATL08_AGG_CONFIG

# Import functions from the utils
from phoreal.io import readers, processing
from phoreal import analysis
from phoreal import plotting as plotter
from phoreal.als.swath import create_als_swath, transform_als_swath
from phoreal.geo.datum_transforms import convert_3d_nad83_to_wgs84
from phoreal.geo.datum_transforms import get_geoid_height

###
import geopandas as gpd
from shapely.geometry import box
import fiona  # <-- 1. Import fiona directly

# Define ATL03 File
aoi = 'Finland'
atl03_dir = '/path/to/home/Downloads'
#atl03_name = 'ATL03_20230623222806_00572006_007_01.h5' # Maine
# atl03_name = 'ATL03_20200212094736_07280607_007_01.h5' # Flordia
# atl03_name = 'ATL03_20220921231443_00191701_007_01.h5' #PuertoRico1
# atl03_name = 'ATL03_20220327195218_00721507_007_01.h5' #PuertoRico2
# atl03_name = 'ATL03_20210801001300_05861203_007_01.h5' #Alaska
# atl03_name = 'ATL03_20190518130651_07680302_007_01.h5' #El Dorado National Forest
# atl03_name = 'ATL03_20191014055846_02650502_007_01.h5' # San Joaquin
# atl03_name = 'ATL03_20200803014101_05910806_007_01.h5' # GRSM
# atl03_name = 'ATL03_20221007064520_02521713_007_01.h5' # New Zealand
atl03_name = 'ATL03_20190516155132_07390305_007_01.h5 ' # Finland

atl03_file = os.path.join(atl03_dir, atl03_name)

# Define ATL08 File
atl08_dir = '/path/to/home/Downloads'
#atl08_name = 'ATL08_20230623222806_00572006_007_01.h5' # Maine 
# atl08_name = 'ATL08_20200212094736_07280607_007_01.h5' # Flordia
# atl08_name = 'ATL08_20220921231443_00191701_007_01.h5' #PuertoRico1
# atl08_name = 'ATL08_20220327195218_00721507_007_01.h5' #PuertoRico2
# atl08_name = 'ATL08_20210801001300_05861203_007_01.h5' #Alaska
# atl08_name = 'ATL08_20190518130651_07680302_007_01.h5' #El Dorado National Forest
# atl08_name = 'ATL08_20191014055846_02650502_007_01.h5' # San Joaquin
# atl08_name = 'ATL08_20200803014101_05910806_007_01.h5' # GRSM
# atl08_name = 'ATL08_20221007064520_02521713_007_01.h5' # New Zealand
atl08_name = 'ATL08_20190913100721_11810405_007_01.h5' # Finland
atl08_file = os.path.join(atl08_dir, atl08_name)

# Define the ground track of interest
gt = 'gt2l'



# Get photon rate DF
df_ph = readers.read_photon_dataframe(atl03_file, gt, atl08_file)

# Filter by extent


# 173.0477,-40.4908
# 168.3525,-46.7752

# extent = np.array([-69.93801,  44.28262, -69.34454,  45.26122]) # Maine
# extent = np.array([-80.99840,24.69424,-80.2819, 26.7259]) # Flordia
# extent = np.array([-145.8605,63.7698,-145.1345,64.8715]) # Alaska
# extent = np.array([-120.5520,37.0259, -119.75960,39.93987]) # El Dorado National Forest
# extent = np.array([-119.87667,36.63366, -118.78267,37.85414]) # San Joaquin
# extent = np.array([-84.4192,34.2622,-81.5986,36.5123]) # GRSM
# extent = np.array([168.3525,-46.7752,173.0477,-40.4908]) # New Zealand
extent = np.array([19.13,59.50, 31.59,70.09]) # Finland

df_ph = processing.filter_df_by_extent(df_ph, extent)
print("******************************")
print(f"Solar elevation: {np.median(df_ph.solar_elevation)}")
print(df_ph.attrs)
print("******************************")

# Plot alongtrack
import matplotlib.pyplot as plt
title = f"{aoi}: {atl03_name} {gt}"
plotter.plot_alongtrack_atl08(df_ph.alongtrack, df_ph.h_ph, df_ph.atl08_class,title)