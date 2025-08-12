#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 15:16:35 2025

@author: ejg2736
"""

import numpy as np

def convert_wgs84_to_nad83_manual(lon, lat, height):
    """
    Converts WGS84 to NAD83 by manually implementing the 7-parameter Helmert transformation.
    This demonstrates the underlying mathematics of a datum shift.

    Args:
        lon (float or np.ndarray): Longitude(s) in WGS84 decimal degrees.
        lat (float or np.ndarray): Latitude(s) in WGS84 decimal degrees.
        height (float or np.ndarray): Ellipsoidal height(s) in WGS84, in meters.

    Returns:
        tuple: A tuple containing (lon_nad83, lat_nad83, height_nad83).
    """
    # WGS84 Ellipsoid parameters
    a_wgs84 = 6378137.0
    f_wgs84 = 1 / 298.257223563
    e2_wgs84 = 2 * f_wgs84 - f_wgs84**2

    # GRS80 Ellipsoid parameters (used by NAD83)
    a_nad83 = 6378137.0
    f_nad83 = 1 / 298.257222101
    e2_nad83 = 2 * f_nad83 - f_nad83**2

    # 7-parameter Helmert transformation (WGS84 to NAD83 for North America)
    tx, ty, tz = -0.991, 1.9072, 0.5129
    rx, ry, rz = 0.02579, 0.00965, 0.01166
    s = -0.00062

    # Convert Geodetic (lon, lat, h) to Cartesian (X, Y, Z) for WGS84
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(lat_rad)**2)
    X_wgs = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    Y_wgs = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    Z_wgs = (N * (1 - e2_wgs84) + height) * np.sin(lat_rad)

    # Apply the 7-parameter Helmert Transformation
    rx_rad, ry_rad, rz_rad = np.radians(rx/3600), np.radians(ry/3600), np.radians(rz/3600)
    scale_factor = s / 1_000_000 + 1
    X_nad = tx + scale_factor * (X_wgs - rz_rad * Y_wgs + ry_rad * Z_wgs)
    Y_nad = ty + scale_factor * (rz_rad * X_wgs + Y_wgs - rx_rad * Z_wgs)
    Z_nad = tz + scale_factor * (-ry_rad * X_wgs + rx_rad * Y_wgs + Z_wgs)

    # Convert Cartesian (X, Y, Z) back to Geodetic (lon, lat, h) for NAD83
    p = np.sqrt(X_nad**2 + Y_nad**2)
    lon_nad_rad = np.arctan2(Y_nad, X_nad)
    lat_nad_rad = np.arctan2(Z_nad, p * (1 - e2_nad83))
    for _ in range(5):
        N_nad = a_nad83 / np.sqrt(1 - e2_nad83 * np.sin(lat_nad_rad)**2)
        height_nad = p / np.cos(lat_nad_rad) - N_nad
        lat_nad_rad = np.arctan2(Z_nad, p * (1 - e2_nad83 * N_nad / (N_nad + height_nad)))
    
    return np.degrees(lon_nad_rad), np.degrees(lat_nad_rad), height_nad


def convert_nad83_to_wgs84_manual(lon, lat, height):
    """
    Converts NAD83 to WGS84 by manually implementing the inverse 7-parameter Helmert transformation.
    This demonstrates the underlying mathematics of a datum shift.

    Args:
        lon (float or np.ndarray): Longitude(s) in NAD83 decimal degrees.
        lat (float or np.ndarray): Latitude(s) in NAD83 decimal degrees.
        height (float or np.ndarray): Ellipsoidal height(s) in NAD83, in meters.

    Returns:
        tuple: A tuple containing (lon_wgs84, lat_wgs84, height_wgs84).
    """
    # Ellipsoid parameters are the same as the forward function
    a_wgs84 = 6378137.0
    f_wgs84 = 1 / 298.257223563
    e2_wgs84 = 2 * f_wgs84 - f_wgs84**2
    a_nad83 = 6378137.0
    f_nad83 = 1 / 298.257222101
    e2_nad83 = 2 * f_nad83 - f_nad83**2

    # Inverse 7-parameter Helmert transformation (NAD83 to WGS84)
    # Note that the signs of all parameters are reversed.
    tx, ty, tz = 0.991, -1.9072, -0.5129
    rx, ry, rz = -0.02579, -0.00965, -0.01166
    s = 0.00062

    # Convert Geodetic (lon, lat, h) to Cartesian (X, Y, Z) for NAD83
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    N = a_nad83 / np.sqrt(1 - e2_nad83 * np.sin(lat_rad)**2)
    X_nad = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    Y_nad = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    Z_nad = (N * (1 - e2_nad83) + height) * np.sin(lat_rad)

    # Apply the inverse 7-parameter Helmert Transformation
    rx_rad, ry_rad, rz_rad = np.radians(rx/3600), np.radians(ry/3600), np.radians(rz/3600)
    scale_factor = s / 1_000_000 + 1
    X_wgs = tx + scale_factor * (X_nad - rz_rad * Y_nad + ry_rad * Z_nad)
    Y_wgs = ty + scale_factor * (rz_rad * X_nad + Y_nad - rx_rad * Z_nad)
    Z_wgs = tz + scale_factor * (-ry_rad * X_nad + rx_rad * Y_nad + Z_nad)

    # Convert Cartesian (X, Y, Z) back to Geodetic (lon, lat, h) for WGS84
    p = np.sqrt(X_wgs**2 + Y_wgs**2)
    lon_wgs_rad = np.arctan2(Y_wgs, X_wgs)
    lat_wgs_rad = np.arctan2(Z_wgs, p * (1 - e2_wgs84))
    for _ in range(5):
        N_wgs = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(lat_wgs_rad)**2)
        height_wgs = p / np.cos(lat_wgs_rad) - N_wgs
        lat_wgs_rad = np.arctan2(Z_wgs, p * (1 - e2_wgs84 * N_wgs / (N_wgs + height_wgs)))
        
    return np.degrees(lon_wgs_rad), np.degrees(lat_wgs_rad), height_wgs

if __name__ == "__main__":
    lon = -80.61936094726622
    lat = 25.179820370629987
    height = 0
    
    nad83_lon, nad83_lat, nad83_height = convert_nad83_to_wgs84_manual(lon, lat, height)
    wgs84_lon, wgs84_lat, wgs84_height = convert_wgs84_to_nad83_manual(lon, lat, height)