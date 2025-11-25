import laspy
import numpy as np
import sys
import os
import geopandas as gpd
import pandas as pd
import shapely.geometry as shp
from shapely.geometry import Point
from shapely.geometry import Polygon
import rasterio
import glob
from pyproj import Transformer, CRS
from tqdm import tqdm
import multiprocessing
# from utils.geographic_utils import utm_zone_epsg, get_geoid_height

def combine_dicts(dicts):
  """Combines multiple dictionaries into a single dictionary.

  Args:
    dicts: A list of dictionaries.

  Returns:
    A combined dictionary.
  """

  result = {}
  for d in dicts:
    result.update(d)
  return result

def find_files(directory, extension, recursively=True):
    if type(extension) == str:
        return glob.glob(f"{directory}/**/*.{extension}",recursive=recursively)
    elif type(extension) == list:
        all_files = []
        for ext in extension:
            files = glob.glob(f"{directory}/**/*.{ext}",recursive=recursively)
            all_files.extend(files)
        return all_files

def read_las_header(las_file):
    # Read file header
    las = laspy.open(las_file) #Using laspy.open('f.las') reads only the header
    
    # Grab useful info
    x_min = las.header.x_min
    x_max = las.header.x_max
    y_min = las.header.y_min
    y_max = las.header.y_max
    z_min = las.header.z_min
    z_max = las.header.z_max
    creatation_date = las.header.creation_date
    parse_crs = las.header.parse_crs() 
    # print('2\n')
    # Read into dictionary
    las_dict = {"file_name":las_file,
            "x_min":x_min,
            "x_max":x_max,
            "y_min":y_min,
            "y_max":y_max,
            "z_min":z_min,
            "z_max":z_max,
            "creation_date":creatation_date,
            "parse_crs":parse_crs}
    # print('read_las_header return\n')
    las.close()

    return las_dict

def process_las_file(las_file):
    """Processes a LAS file and returns its header data."""
    try:
        return read_las_header(las_file)
    except Exception as e:
        print(f"\nError processing {las_file}: {e}\n")
        return None

# def start_process():
#     print('Starting', multiprocessing.current_process().name)

def process_las_files_multiprocessing(las_list, num_processes=None):
    """Processes a list of LAS files using multiprocessing."""
    if num_processes is None:
        num_processes = np.ceil(multiprocessing.cpu_count()/2).astype(int)
    print('Reading header files:')
    pool = multiprocessing.Pool(processes=3)
    results = tqdm(pool.imap_unordered(process_las_file, las_list),
                   total = len(las_list))

    data = [result for result in results if result is not None]
    pool.close()
    return data

def create_gdf_las_extent(las_list, num_processes=None):
    """
    Creates a GeoPandas DataFrame containing geometries for the extent of LAS files.
    
    Args:
        las_list (list): List of paths to LAS files.
        num_processes (int, optional): Number of processes to use for multiprocessing.
            Defaults to None (use single process).
    
    Returns:
        geopandas.GeoDataFrame: GeoPandas DataFrame with geometries and CRS.
    """
    
    # Process LAS files (assuming process_las_files_multiprocessing exists)
    data = process_las_files_multiprocessing(las_list,
                                             num_processes=num_processes)
    df = pd.DataFrame(data)
    
    # Handle rows with missing CRS
    missing_crs = df[df['parse_crs'].isnull()]
    if len(missing_crs) > 0:
        print(f"Warning: {len(missing_crs)}/{len(df)} LAS files have missing CRS. Removing them.")
        df = df.dropna(subset=['parse_crs'])
      
    # Check for multiple CRS (if relevant)
    crs_list = df['parse_crs'].unique()
    if len(crs_list) > 1:
        print(f"Warning: There are {len(crs_list)} unique CRS in dataset")
        
    # Initialize empty lat/lon columns with zeros
    df['lat_min'] = np.zeros(len(df))
    df['lat_max'] = np.zeros(len(df))
    df['lon_min'] = np.zeros(len(df))
    df['lon_max'] = np.zeros(len(df))
    
    # Iterate through unique CRS and reproject extents
    for crs in df['parse_crs'].unique():
        transformer = Transformer.from_crs(crs.to_2d(), 4326)
    
        df.loc[df['parse_crs'] == crs, 'lat_min'], df.loc[df['parse_crs'] == crs, 'lon_min'] = transformer.transform(
            df.loc[df['parse_crs'] == crs, 'x_min'], df.loc[df['parse_crs'] == crs, 'y_min'])
    
        df.loc[df['parse_crs'] == crs, 'lat_max'], df.loc[df['parse_crs'] == crs, 'lon_max'] = transformer.transform(
            df.loc[df['parse_crs'] == crs, 'x_max'], df.loc[df['parse_crs'] == crs, 'y_max'])

    # Create geometries from lat/lon extents using Shapely
    df['geometry'] = df.apply(lambda row: shp.box(row['lon_min'], 
                                                  row['lat_min'], 
                                                  row['lon_max'], 
                                                  row['lat_max']), axis=1)
    
    # Create GeoPandas DataFrame
    wgs_extent = gpd.GeoDataFrame(df,geometry='geometry',crs=CRS.from_epsg(4326))
    
    return wgs_extent


def process_geotiff(filepath):
    """Processes a single GeoTIFF file and returns its extent."""
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            polygon = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.right, bounds.bottom),
                (bounds.right, bounds.top),
                (bounds.left, bounds.top),
                (bounds.left, bounds.bottom),
            ])
            return polygon, os.path.basename(filepath), src.crs
    except rasterio.errors.RasterioIOError as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None

def geotiff_extents_to_shapefile_parallel(folder_path, output_shapefile):
    """
    Creates a shapefile containing the extents of all GeoTIFFs in a folder using multiprocessing and tqdm.

    Args:
        folder_path (str): Path to the folder containing GeoTIFF files.
        output_shapefile (str): Path to the output shapefile.
    """

    geotiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".tif")]

    if not geotiff_files:
        print("No valid GeoTIFFs found in the folder.")
        return

    with multiprocessing.Pool(10) as pool:
        results = list(tqdm(pool.imap(process_geotiff, geotiff_files), total=len(geotiff_files), desc="Processing GeoTIFFs"))

    geometries = []
    filenames = []
    crs = None

    for polygon, filename, file_crs in results:
        if polygon:
            geometries.append(polygon)
            filenames.append(os.path.join(folder_path,filename))
            if crs is None and file_crs is not None:
              crs = file_crs

    if not geometries:
        print("No valid geometries were created.")
        return

    gdf = gpd.GeoDataFrame({'file_name': filenames, 'geometry': geometries}, crs=crs)

    gdf.to_file(output_shapefile, driver='GPKG')
    print(f"Shapefile created: {output_shapefile}")

def get_als_from_target(wgs_extent, target_lon, target_lat, region = None):
    print('Region ' + region)
    target_point = Point(target_lon, target_lat)
    distance = 0.0001 * 2 # Define footprint length*2
    point_buffer = target_point.buffer(distance)

    polygons_within_distance = wgs_extent[wgs_extent.intersects(point_buffer)]
    
    if len(polygons_within_distance) == 0:
        print('Warning: No ALS tile identified for intersection')
        return pd.DataFrame()
    elif len(polygons_within_distance) > 1:
        print('Warning: Multiple ALS tiles identified for intersection')
        
    # Convert target location
    if region == 'Finland':
        print('Warning: Target ALS points done in meters')
        footprint_radius = 6
    else:
        print('Warning: Target ALS points done in feet')
        footprint_radius = 6 * 3.28 # Approx in Feet
    
    #WARNING
    transformer = Transformer.from_crs(4326, 
                                    polygons_within_distance.iloc[0].parse_crs)
    
    target_northing, target_easting = transformer.transform(target_lat, 
                                                            target_lon)
    data_list = []
    for i in range(0,len(polygons_within_distance)):
        las_file = laspy.read(polygons_within_distance.iloc[i].file_name)
        x = las_file.x
        y = las_file.y
        z = las_file.z
        c = las_file.classification
        
        # x = x[c == 2]
        # y = y[c == 2]
        # z = z[c == 2]
        # c = c[c == 2]

        # x, y, z, c = read_and_reproject_las(
        #     polygons_within_distance.iloc[i].file_name)
        
        dist = np.sqrt(np.array(x - target_northing)**2 +\
                       np.array(y - target_easting)**2)
        
        #if len(x[dist < footprint_radius]) == 0:
        #    las_dict = {}
        #    break

        las_dict = {'x':np.array(x[dist < footprint_radius]),
                'y':np.array(y[dist < footprint_radius]),
                'z':np.array(z[dist < footprint_radius]),
                'c':np.array(c[dist < footprint_radius])
                }
        
        if len(las_dict['x']) > 0:
            data_list.append(las_dict)
            
    data_dict = combine_dicts(data_list)
    
    if len(data_dict) == 0:
        print('Warning: No ALS data found at crossover target')
        return pd.DataFrame()

    if region == 'Finland':
        print('Warning: Target EPSG Code Hard Coded')
        crs_las = 'EPSG:32634'
        
    else: 
        print('Warning: EPSG Code automatically chooses from pre-selected list')
        crs_las = CRS.from_user_input(polygons_within_distance.iloc[0].parse_crs)
    
        if 'California zone 1' in str(crs_las):
            crs_las = 'EPSG:8714'
        elif 'California zone 2' in str(crs_las):
            crs_las = 'EPSG:8715'
        elif 'California zone 3' in str(crs_las):
            crs_las = 'EPSG:8716'
        elif 'California zone 4' in str(crs_las):
            crs_las = 'EPSG:8717'  
        elif 'California zone 5' in str(crs_las):
            crs_las = 'EPSG:8718'  
            
        elif 'California zone 6' in str(crs_las):
            crs_las = 'EPSG:8719'
        elif '6543' in str(crs_las):
            crs_las = 'EPSG:6543'
        else:
            print('None can be found, using crs coords in las file')
        
    #target_epsg = utm_zone_epsg(target_lat, target_lon)
    if region == 'SJ':
        target_epsg = 'EPSG:32610'
    elif region == 'NC':
        target_epsg = 'EPSG:32617' # North Carolina
    elif region == 'Finland':
        target_epsg = 'EPSG:32634' # Finland
    
    try_3d = True
    if try_3d:
        if region == 'SJ':
            geoid_file = '/home/ejg2736/data/geoid/BundleAll/g2018_conus.gtx'
            data_dict['x'],data_dict['y'],data_dict['z'] = \
                reproject_xyz(data_dict['x'],data_dict['y'],data_dict['z'],
                              crs_las,target_epsg, geoid_file)
        elif region == 'NC':
            crs_las = 'EPSG:6543'
            data_dict['z'] = data_dict['z'] * 0.3043
            geoid_file = '/home/ejg2736/data/geoid/BundleAll/g2012a_conus.gtx'
            data_dict['x'],data_dict['y'],data_dict['z'] = \
                reproject_xyz(data_dict['x'],data_dict['y'],data_dict['z'],
                              crs_las,target_epsg, geoid_file)
        elif region == 'Finland':
            geoid_file = '/home/ejg2736/data/geoid/BundleAll/egm08_1.gtx'
            data_dict['x'],data_dict['y'],data_dict['z'] = \
                reproject_xyz(data_dict['x'],data_dict['y'],data_dict['z'],
                              crs_las,target_epsg, geoid_file)
                
    else:
        transformer = Transformer.from_crs(CRS(crs_las),CRS(target_epsg))
        data_dict['x'], data_dict['y'] = transformer.transform(data_dict['x'],data_dict['y'])
        
    df_als = pd.DataFrame(data_dict)
    
    return df_als

def reproject_xyz(x,y,z,epsg_in,epsg_out, geoid_file):
    if isinstance(epsg_in,str):
        # geoid_file = '/home/ejg2736/data/geoid/BundleAll/g2018_conus.gtx' # SJ
        # geoid_file = '/home/ejg2736/data/geoid/BundleAll/g2012a_conus.gtx' # NC
        # geoid_file = '/home/ejg2736/data/geoid/BundleAll/egm08_1.gtx' # Finland
        
        print('Warning: Hard coded is reference geoid')
        epsg_in = CRS(init=epsg_in,geoidgrids=geoid_file)
    
        transformer_3d = Transformer.from_crs(epsg_in.to_3d(),
                                              CRS(epsg_out).to_3d(),
                                              always_xy=True)

        #print('Warning: Hard coded vertical feet to meter')        
        #z = z * 0.3043
        
        x, y, z = transformer_3d.transform(x,y,z)

    else:

        transformer_3d = Transformer.from_crs(epsg_in,
                                              CRS(epsg_out).to_3d(),
                                              always_xy=True)
        
        x, y, z = transformer_3d.transform(x,y,z)


    return x, y, z
  
def read_and_reproject_las(las_file):
    geoid_file = '/home/ejg2736/data/geoid/BundleAll/g2018_conus.gtx'

    las = laspy.read(las_file)
    
    if las.header.parse_crs().to_epsg() is not None:
        las_crs_3d = las.header.parse_crs()
    else:        
        las_crs_3d = CRS(init='EPSG:8716',geoidgrids=geoid_file)
    
    transformer_3d = Transformer.from_crs(las_crs_3d.to_3d(),
                                          CRS("EPSG:32610").to_3d(),
                                          always_xy=True)

    x, y, z = transformer_3d.transform(las.x,las.y,las.z)
    
    return x, y, z, las.classification

if __name__ == "__main__":
    # las_file = '/home/ejg2736/data/cross_over/als/san_joaquin1/Job1042832_37121_40_06.laz'
    # las_file = '/home/ejg2736/data/als_test_area/USGS_LPC_CA_SanJoaquin_2021_A21_s62225w21125.laz'



    # Test search
    # las_dir = '/home/ejg2736/data/cross_over/als'
    # las_dir = '/home/ejg2736/data/als_test_area'
    # las_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/CA_SanJoaquin'
    # las_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/Finland/UTM/'
    # las_list = find_files(las_dir,'laz')
    # wgs_extent = create_gdf_las_extent(las_list, num_processes=5)
    # target_lon = -121.15049352248671
    # target_lat = 37.42051698302789
    # target_lon = -121.66523557936257
    # target_lat = 37.797649318848
    # target_lon = -121.66523557936000
    # target_lat = 37.797649318000
    

    # # # Convert coords to same coordinate system
    # import time
    # t1 = time.time()

    # df_als = get_als_from_target(wgs_extent, target_lon, target_lat)
    # t2 = time.time()
    # print(t2-t1)
    
    # Example usage:
    # folder_path = "/home/ejg2736/Desktop/Bigtex/exports/vol2/vol2/Data/OpenData/Finland/UTM_DTM"  # Replace with your folder path
    # output_shapefile = "geotif_extents.shp"  # Replace with your desired output path
    # output_shapefile = "geotif_extents.gpkg"  # Replace with your desired output path
    
    folder_path = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/Global/fabdem/zip_tiles'
    output_shapefile = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/Global/fabdem/fabdem_tile_extents3.gpkg'

    geotiff_extents_to_shapefile_parallel(folder_path, output_shapefile)
    
    geotiff_gdf = gpd.read_file(output_shapefile)
