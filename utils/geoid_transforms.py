import math
# import pyproj
import rasterio
import numpy as np
from utils.datum_transforms import convert_3d_nad83_to_wgs84


def get_geoid_height(lon, lat, geoid_file):
  """Retrieves geoid height for a given longitude and latitude.

  Args:
    lon: Longitude of the point.
    lat: Latitude of the point.
    geoid_file: Path to the geoid model file.

  Returns:
    Geoid height at the specified location, or None if the location is outside the geoid model's coverage.
  """
  lon = np.array(lon)
  lat = np.array(lat)

  with rasterio.open(geoid_file) as src:
    geoid_data = src.read(1)
    transform = src.transform
    if src.bounds[2] > 185:
        lon = lon + 360
        
        
    col, row = ~transform * (lon, lat)
    
    # if 0 <= row < src.height and 0 <= col < src.width:
    if (type(lon) == np.ndarray) and (type(lat) == np.ndarray):
        return geoid_data[row.astype(int).tolist(), col.astype(int).tolist()]
    elif (type(lon) == float) and (type(lat) == float):
        return geoid_data[int(row), int(col)]
