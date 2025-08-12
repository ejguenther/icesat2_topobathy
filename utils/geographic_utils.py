import math
# import pyproj
import rasterio
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates an estimated distance between two points on Earth.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:  
        float: Distance between the two points in meters.  
    """

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon  
        / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1  
 - a))
    r = 6371000  # Approx. Earth radius in meters
    distance = c * r

    return distance


def find_utm_zone(lat, lon):
    """
    Determines the UTM zone for a given latitude and longitude.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        tuple: (zone_number, zone_letter)
    """

    zone_number = int((lon + 180) / 6) + 1

    if lat >= 84:
        zone_letter = 'X'
    elif lat < -80:
        zone_letter = 'C'
    else:
        zone_letter = chr(int((lat + 80) / 8) + ord('C'))

    # Special case for Norway and Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32

    return zone_number, zone_letter

def find_utm_zone_epsg(lat, lon):
    """
    Determines the UTM zone for a given latitude and longitude.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        string: "EPSG:XXXX"
    """

    zone_number = int((lon + 180) / 6) + 1

    if lat >= 84:
        return 'EPSG:32661'
    elif lat < -80:
        return 'EPSG:32761'
    elif lat > 0:
        epsg_code = 'EPSG:326'
    else:
        epsg_code = 'EPSG:327'
    
    epsg_code = epsg_code + str(zone_number).zfill(2)
    
    # Special case for Norway and Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        epsg_code = 'EPSG:32632'

    return epsg_code




def get_geoid_height(lon, lat, geoid_file):
  """Retrieves geoid height for a given longitude and latitude.

  Args:
    lon: Longitude of the point.
    lat: Latitude of the point.
    geoid_file: Path to the geoid model file.

  Returns:
    Geoid height at the specified location, or None if the location is outside the geoid model's coverage.
  """

  with rasterio.open(geoid_file) as src:
    geoid_data = src.read(1)
    transform = src.transform
        
    col, row = ~transform * (lon, lat)
    
    # if 0 <= row < src.height and 0 <= col < src.width:
    if (type(lon) == np.ndarray) and (type(lat) == np.ndarray):
        return geoid_data[row.astype(int).tolist(), col.astype(int).tolist()]
    elif (type(lon) == float) and (type(lat) == float):
        return geoid_data[int(row), int(col)]
