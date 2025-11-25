import math
# import pyproj
import rasterio
import numpy as np
from geopy.distance import geodesic
import pandas as pd

def calculate_distance(row):
    # Handle the first row (which has no previous point)
    if pd.isna(row['lat_prev']):
        return 0.0
    
    # Create the coordinate tuples geopy expects
    point_current = (row['_lat'], row['_lon'])
    point_previous = (row['lat_prev'], row['lon_prev'])
    
    # Calculate and return the distance
    return geodesic(point_previous, point_current).meters

def get_df_distance(df,lat_field='latitude',lon_field='longitude',out_field = 'alongtrack_test'):
    df = df.rename(columns={lat_field:'_lat',lon_field:'_lon'})
    df['lat_prev'] = df['_lat'].shift(1)
    df['lon_prev'] = df['_lon'].shift(1)
    df['distance_meters'] = df.apply(calculate_distance, axis=1)
    df = df.drop(columns=['lat_prev', 'lon_prev'])
    df = df.rename(columns={'_lat':lat_field,'_lon':lon_field})
    return df


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




def vincenty_inverse(lat1, lon1, lat2, lon2, max_iter=200, tol=1e-12):
    """
    Calculates the geodesic distance between two points on the WGS 84
    ellipsoid using Vincenty's inverse formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Tolerance for convergence.

    Returns:
        float: Distance between the two points in meters.
                Returns float('nan') if it fails to converge.
    """
    
    # WGS 84 ellipsoid parameters (from your request)
    a = 6378137.0         # Equatorial radius in meters
    b = 6356752.314245    # Polar radius in meters (using standard WGS 84 value for more precision)
    f = (a - b) / a       # Flattening
    
    # Convert degrees to radians
    phi_1 = math.radians(lat1)
    lambda_1 = math.radians(lon1)
    phi_2 = math.radians(lat2)
    lambda_2 = math.radians(lon2)

    # Reduced latitude (latitude on the auxiliary sphere)
    U1 = math.atan((1 - f) * math.tan(phi_1))
    U2 = math.atan((1 - f) * math.tan(phi_2))
    
    sin_U1 = math.sin(U1)
    cos_U1 = math.cos(U1)
    sin_U2 = math.sin(U2)
    cos_U2 = math.cos(U2)

    L = lambda_2 - lambda_1 # Difference in longitude
    lambda_ = L             # Iteration variable
    lambda_prev = 0

    for _ in range(max_iter):
        sin_lambda = math.sin(lambda_)
        cos_lambda = math.cos(lambda_)
        
        sin_sigma = math.sqrt((cos_U2 * sin_lambda) ** 2 + \
                              (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2)
        
        if sin_sigma == 0:
            return 0.0  # Co-incident points

        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        
        sin_alpha = (cos_U1 * cos_U2 * sin_lambda) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2

        # Handle equatorial line
        if cos_sq_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - (2 * sin_U1 * sin_U2) / cos_sq_alpha

        C = (f / 16) * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        
        lambda_prev = lambda_
        lambda_ = L + (1 - C) * f * sin_alpha * \
                    (sigma + C * sin_sigma * \
                    (cos_2sigma_m + C * cos_sigma * \
                    (-1 + 2 * cos_2sigma_m ** 2)))

        if abs(lambda_ - lambda_prev) < tol:
            break # Convergence
    else:
        return float('nan') # Failed to converge

    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + (u_sq / 16384) * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = (u_sq / 1024) * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    
    delta_sigma = B * sin_sigma * (cos_2sigma_m + (B / 4) * \
                   (cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) - \
                   (B / 6) * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * \
                   (-3 + 4 * cos_2sigma_m ** 2)))
    
    s = b * A * (sigma - delta_sigma) # Final distance
    
    return s


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
