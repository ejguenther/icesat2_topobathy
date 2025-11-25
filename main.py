#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:54:03 2025

@author: ejg2736
"""

import h5py
import os
import numpy as np
from utils import processing, analysis
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from utils.geographic_utils import find_utm_zone_epsg, get_geoid_height
from utils.datum_transforms import convert_3d_nad83_to_wgs84
from utils.create_las_swath import create_als_swath

from pyproj import Transformer

def t2t(hours, mins=None, sec=None):
    """
    This function changes hms into seconds, or 
    HH:MM:SSSSSS format, such as in GPS time-stamps.
    """
    if mins==None and sec== None:
        tot_sec = hours%86400
        choice = 1
    elif hours != None and mins != None and sec != None:
        choice = 2
        if hours > 24 or mins > 60 or sec > 60 or hours < 0 or mins < 0 or sec < 0:
            raise ValueError("Value out of range")
    if choice == 1:
        hours = int(tot_sec/3600)
        mins = int((tot_sec%3600)/(60))
        sec = ((tot_sec%3600)%(60))

    if choice == 2:
        tot_sec = hours*3600 + mins*60 + sec

    return [tot_sec, "{:>02d}:{:>02d}:{:>09.6f}".format(hours, mins, sec)]

def get_date(*args, debug=0):
    
    """
    Example:
    import icesatUtils
    doy = icesatUtils.get_date(year, month, day)
    # or
    month, day = icesatUtils.get_date(year, doy)

    """
    def help():
        print("""
    Example:
    import icesatUtils
    doy = icesatUtils.get_date(year, month, day)
    # or
    month, day = icesatUtils.get_date(year, doy)
            """)

    import datetime

    if len(args) == 2:
        y = int(args[0]) #int(sys.argv[1])
        d = int(args[1]) #int(sys.argv[2])
        yp1 = datetime.datetime(y+1, 1, 1)
        y_last_day = yp1 - datetime.timedelta(days=1)
        if d <= y_last_day.timetuple().tm_yday:
            date = datetime.datetime(y, 1, 1) + datetime.timedelta(days=d-1)
            return date.month, date.day
        else:
            print("error")
            help()

    elif len(args) == 3:
        y, m, d = args
        date = datetime.datetime(y, m, d)
        doy = int(date.timetuple().tm_yday)
        # print("doy = {}".format(date.timetuple().tm_yday))
        return str(doy).zfill(3)

    else:
        print("error: incorrect number of args")
        help()

def get_h5_meta(h5_file, meta='date', rtn_doy=False, rtn_hms=True, file_start='ATL'): 
    """
    This function gets metadata directly from the ATL filename.

    Input:
        h5_file - the ATL file, full-path or not
        meta - the type of metadata to output
            ['date', 'track', 'release', 'version', 
                'f_type', 'hms', 'cycle']
            f_type - either rapid or final
        rtn_doy - return day of year or not, if date is chosen
        rtn_hms - return hour/min/sec, or time in sec
        file_start - search for info based on file_start index;
                        useful if given an ATL file that starts
                        with "errorprocessing_..." or any other
                        prefix
        debug - for small errors

    Output:
        varies, but generally it's one value, unless 'hms' meta is chosen,
        in which case it is two values.

    Example:
        import icesatUtils
        fn = DIR + '/ATL03_20181016000635_02650109_200_01.h5'
        # or fn = 'ATL03_20181016000635_02650109_200_01.h5'
        year, day_of_year = icesatUtils.get_h5_meta(fn, meta='date', rtn_doy=True)
        version = icesatUtils.get_h5_meta(fn, meta='version')
        release = icesatUtils.get_h5_meta(fn, meta='release')
    """

    h5_file = os.path.basename(h5_file) # h5_file.split('/')[-1]

    meta = meta.lower()

    i0 = 0
    try:
        i0 = h5_file.index(file_start)
    except ValueError:
        print('warning: substring %s not found in %s' % (file_start, h5_file))
    # i0 = check_file_start(h5_file, file_start, debug)

    if meta == 'date':
        year = int(h5_file[i0+6:i0+10])
        month = int(h5_file[i0+10:i0+12])
        day = int(h5_file[i0+12:i0+14])

        if rtn_doy:
            doy0 = get_date(year, month, day)
            return str(year), str(doy0).zfill(3)

        return str(year), str(month).zfill(2), str(day).zfill(2)

    elif meta == 'track':
        return int(h5_file[i0+21:i0+25])

    elif meta == 'release':
        r = h5_file[i0+30:i0+34]
        if '_' in r:
            r = h5_file[i0+30:i0+33]
        return r

    elif meta == 'version':
        v = h5_file[i0+34:i0+36]
        if '_' in v:
          v = h5_file[i0+35:i0+37]
        return v

    elif meta == 'f_type':
        r = h5_file[i0+30:i0+34]
        f_type = 'rapid'
        if '_' in r:
            r = h5_file[i0+30:i0+33]
            f_type = 'final'
        return f_type

    elif meta == 'hms':
        hms = h5_file[i0+14:i0+20]
        h, m, s = hms[0:2], hms[2:4], hms[4:6]
        if rtn_hms:
            return h, m, s
        else:
            h, m, s = int(h), int(m), int(s)
            t0, t0_full = t2t(h,m,s)
            return t0, t0_full

    elif meta == 'cycle':
        return int(h5_file[i0+25:i0+29])

    else:
        print('error: unknown meta=%s' % meta)
        return 0

def get_attribute_info(atlfilepath, gt):
    # add year/doy, sc_orient, beam_number/type to 08 dataframe
    year, doy = get_h5_meta(atlfilepath, meta='date', rtn_doy=True)

    with h5py.File(atlfilepath, 'r') as fp:
        try:
            fp_a = fp[gt].attrs
            description = (fp_a['Description']).decode()
            beam_type = (fp_a['atlas_beam_type']).decode()
            atlas_pce = (fp_a['atlas_pce']).decode()
            spot_number = (fp_a['atlas_spot_number']).decode()
            atmosphere_profile = (fp_a['atmosphere_profile']).decode()
            groundtrack_id = (fp_a['groundtrack_id']).decode().lower()
            sc_orient = (fp_a['sc_orientation']).decode().lower()
        except:
            description = ''
            beam_type = ''
            atlas_pce = ''
            spot_number = ''
            atmosphere_profile = ''
            groundtrack_id = ''
            sc_orient = ''
    info_dict = {
        "description" : description,
        "atlas_beam_type" : beam_type,
        "atlas_pce" : atlas_pce,
        "atlas_spot_number" : spot_number,
        'atmosphere_profile' : atmosphere_profile,
        "groundtrack_id" : groundtrack_id,
        "sc_orientation" : sc_orient,
        "year" : year,
        "doy" : doy

        }
    
    return info_dict


def get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt):
    # Read ATL03
    f = h5py.File(atl03_file, 'r') 
    
    # Read photon rate ATL03 data
    lon_ph = np.array(f[gt + '/heights/lon_ph'])
    lat_ph = np.array(f[gt + '/heights/lat_ph'])
    h_ph = np.array(f[gt + '/heights/h_ph'])
    quality_ph = np.array(f[gt + '/heights/quality_ph'])
    delta_time = np.array(f[gt + '/heights/delta_time'])
    
    # Read ATL03 segment rate at ATL03 photon rate
    solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/solar_elevation')
    alongtrack = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/segment_dist_x')
    alongtrack = alongtrack + np.array(f[gt + '/heights/dist_ph_along'])
    
    # Read ATL08 signal photon at ATL03 photon rate
    atl08_class_ph = processing.get_atl08_class_to_atl03(atl03_file, atl08_file,gt)
    atl08_norm_h_ph = processing.get_atl08_norm_h_to_atl03(atl03_file, atl08_file,gt)
    
    
    # Read ATL24 photon rate at ATL03 photon rate
    atl24_class_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt)
    atl24_ortho_h_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt,'/ortho_h')
    atl24_conf = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt,'/confidence')
    
    # Combine ATL08 and ATL4 classifications
    combined_class_ph = processing.combine_atl08_and_atl24_classifications(atl08_class_ph,atl24_class_ph)
    
    # Identify contested photon classifications
    contested_class_ph = processing.identify_contested_photons(atl08_class_ph, atl24_class_ph)

    
    # Create pandas dataframe
    df_ph = pd.DataFrame(
            {
                "alongtrack": alongtrack,
                "latitude":lat_ph,
                "longitude":lon_ph,
                "h_ph": h_ph,
                "h_norm": atl08_norm_h_ph,
                "ortho_h": atl24_ortho_h_ph,
                "atl08_class":atl08_class_ph,
                "atl24_class":atl24_class_ph,
                "atl24_conf":atl24_conf,
                "combined_class":combined_class_ph,
                "contested_class":contested_class_ph,
                "solar_elevation":solar_elevation,
                "quality_ph":quality_ph,
                "delta_time":delta_time
            }
        )
    
    return df_ph

def get_atl_at_seg(df_ph, res = 20, min_at = None):
    if not min_at:
        min_at = np.min(df_ph.alongtrack)
    key = np.floor((df_ph.alongtrack - min_at)/res).astype(int)

    df_ph['key_id'] = key
    df_seg = pd.DataFrame({'key_id':np.unique(key)})
    
    # Calculate alongtrack
    df_seg['alongtrack'] = ((np.unique(key) * res) + min_at) + (res/2)
    
    # Calculate median latitude    
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'latitude',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'latitude'
    )    

    # Calculate median longitude
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'longitude',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'longitude'
    )    

    # Calculate solar_elevation, median solar_elevation (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'solar_elevation',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'solar_elevation'
    )    

    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1],
        outfield = 'h_te_median'
    )    

    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'h_norm',
        operation = 'get_max98',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'h_canopy'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'h_norm',
        operation = 'max',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'h_canopy_max'
    )


    if 'h_topobathy_norm' in df_ph.columns: 
        # Calculate h_canopy, 98th percentile canopy height relative to ground
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_topobathy_norm',
            operation = 'get_max98',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_tb'
        )
        
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_topobathy_norm',
            operation = 'max',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_tb_max'
        )
    
    if 'h_toposurf_norm' in df_ph.columns: 
        # Calculate h_canopy, 98th percentile canopy height relative to ground
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_toposurf_norm',
            operation = 'get_max98',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_ts'
        )
        
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_toposurf_norm',
            operation = 'max',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_ts_max'
        )
    
    if 'h_te_norm' in df_ph.columns: 
        # Calculate h_canopy, 98th percentile canopy height relative to ground
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_te_norm',
            operation = 'get_max98',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_te'
        )
        
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_te_norm',
            operation = 'max',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_te_max'
        )
        
    if 'h_surf_norm' in df_ph.columns: 
        # Calculate h_canopy, 98th percentile canopy height relative to ground
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_surf_norm',
            operation = 'get_max98',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_sf'
        )
        
        df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
            key_field = 'key_id',
            field = 'h_surf_norm',
            operation = 'max',
            class_field = 'atl08_class',
            class_id = [2,3],
            outfield = 'h_canopy_sf_max'
        )
    
    # Calculate h_canopy_abs, 98th percentile canopy height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'max98',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'h_canopy_abs'
    )
    
    # Calculate h_bathy, the median bathymetic height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl24_class',
        class_id = [40],
        outfield = 'h_bathy'
    )
    
    # Calculate h_bathy, the median bathymetic height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'combined_class',
        class_id = [1,40],
        outfield = 'h_topobathy'
    )
    
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl24_class',
        class_id = [41],
        outfield = 'h_surface'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'atl08_class',
        class_id = [1],
        outfield = 'n_terrain'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'n_canopy'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'atl24_class',
        class_id = [40],
        outfield = 'n_bathy'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'atl24_class',
        class_id = [41],
        outfield = 'n_surf'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'atl08_class',
        class_id = [0],
        outfield = 'n_unclass_atl08'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'delta_time',
        operation = 'get_len_unique',
        class_field = 'atl08_class',
        class_id = [0,1,2,3],
        outfield = 'n_shots'
    )
    
    # Filter NaN data
    df_seg.loc[df_seg.longitude == 0,"longitude"] = np.nan
    df_seg.loc[df_seg.latitude == 0,"latitude"] = np.nan
    df_seg.dropna(subset =['longitude','latitude'],inplace=True)

    
    # Filter low canopy out
    canopy_check = df_seg.h_canopy_abs - df_seg.h_surface
    df_seg.loc[canopy_check < 1, "h_canopy"] = np.nan
    
    # Identify composition of each segment
    comp_flag = processing.get_measurement_type_string(df_seg, ['h_te_median','h_canopy',
                                                     'h_bathy','h_surface'])
    df_seg['comp_flag'] = comp_flag
    df_seg['comp_flag'] =(
        df_seg['comp_flag']
        .str.replace('h_te_median', 'terrain')
        .str.replace('h_canopy', 'canopy')
        .str.replace('h_surface', 'surface')
        .str.replace('h_bathy', 'bathymetry')
        )
    


    return df_seg


def get_als_at_seg(als_swath, res = 20, min_at = None, height = 'ellip_h'):
    if not min_at:
        min_at = np.min(als_swath.alongtrack)
    key = np.floor((als_swath.alongtrack - min_at)/res).astype(int)

    als_swath['key_id'] = key
    df_seg = pd.DataFrame({'key_id':np.unique(key)})
    
    # Calculate alongtrack
    df_seg['alongtrack'] = ((np.unique(key) * res) + min_at) + (res/2)
    
    # Calculate median x coord    
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'x',
        operation = 'median',
        class_field = 'classification',
        class_id = list(range(1, 100)),
        outfield = 'x_als'
    )    

    # Calculate median y coord
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'y',
        operation = 'median',
        class_field = 'classification',
        class_id = list(range(1, 100)),
        outfield = 'y_als'
    )    
 

    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'classification',
        class_id = [2],
        outfield = 'als_topo_median'
    )    
    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'classification',
        class_id = [40],
        outfield = 'als_bathy_median'
    )    

    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'classification',
        class_id = [2,40],
        outfield = 'als_topobathy_median'
    )    
    
    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'classification',
        class_id = [41],
        outfield = 'als_surface_median'
    )    


    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_max98',
        class_field = 'classification',
        class_id = [3,4,5],
        outfield = 'als_unclassed_max98'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'h_norm',
        operation = 'get_max98',
        class_field = 'classification',
        class_id = [3,4,5],
        outfield = 'als_norm_veg_max98'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'h_topobathy_norm',
        operation = 'get_max98',
        class_field = 'classification',
        class_id = [3,4,5],
        outfield = 'als_tb_norm_veg_max98'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'classification',
        class_id = [3,4,5],
        outfield = 'als_n_unclass'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'classification',
        class_id = [3,4,5],
        outfield = 'als_n_terrain'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'classification',
        class_id = [2],
        outfield = 'als_n_terrain'
    )
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'classification',
        class_id = [40],
        outfield = 'als_n_bathy'
    )
        
    
    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'get_len',
        class_field = 'classification',
        class_id = [41],
        outfield = 'als_n_surf'
    )
    
    # # Calculate h_canopy_abs, 98th percentile canopy height (orthometric)
    # df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
    #     key_field = 'key_id',
    #     field = 'z',
    #     operation = 'max98',
    #     class_field = 'classification',
    #     class_id = [2,3],
    #     outfield = 'h_canopy_abs'
    # )
    
    # # Filter NaN data
    # df_seg.loc[df_seg.longitude == 0,"longitude"] = np.nan
    # df_seg.loc[df_seg.latitude == 0,"latitude"] = np.nan
    # df_seg.dropna(subset =['longitude','latitude'],inplace=True)

    
    # # Filter low canopy out
    # canopy_check = df_seg.h_canopy_abs - df_seg.h_surface
    # df_seg.loc[canopy_check < 1, "h_canopy"] = np.nan
    
    # Identify composition of each segment
    comp_flag = processing.get_measurement_type_string(df_seg, ['als_topo_median','als_bathy_median',
                                                     'als_surface_median','als_unclassed_max98'])
    df_seg['als_comp_flag'] = comp_flag
    df_seg['als_comp_flag'] =(
        df_seg['als_comp_flag']
        .str.replace('als_topo_median', 'terrain')
        .str.replace('als_unclassed_max98', 'unclassed')
        .str.replace('als_surface_median', 'surface')
        .str.replace('als_bathy_median', 'bathymetry')
        )
    


    return df_seg



def filter_df_by_extent(df, extent):
    minx, miny, maxx, maxy = extent
    filtered_df = df[
    (df['longitude'] >= minx) & (df['longitude'] <= maxx) &
    (df['latitude'] >= miny) & (df['latitude'] <= maxy)
    ]
    
    return filtered_df
    
def compute_metrics(ref, measure):
    err = ref - measure
    err = err.dropna()
    n = len(err)
    mean_err = np.mean(err)
    mae = np.mean(np.abs(err))
    squared_diff = (err)**2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return n, mean_err, mae, mse, rmse
    
def find_corresponding_atl(source_filename, target_dir):
    """
    Finds a corresponding ICESat-2 file in a target directory.

    This function works by extracting the unique date/time and track number
    from the source filename and searching for a file in the target directory
    that contains the same identifier.

    Args:
        source_filename (str): The basename of the source file (e.g., 'ATL24_...h5').
        target_dir (str): The path to the directory to search for the target file.

    Returns:
        str: The full path to the corresponding file if found, otherwise None.
    """
    try:
        # Split the source filename to get the key components.
        # The key is the date/time (index 1) and track number (index 2).
        source_parts = source_filename.split('_')
        file_identifier = f"_{source_parts[1]}_{source_parts[2]}_"
    except IndexError:
        # Handle filenames that don't match the expected ATL format
        print(f"Warning: Could not parse a valid identifier from: {source_filename}")
        return None

    # Use pathlib for robust, cross-platform path handling
    target_path = Path(target_dir)

    # Iterate through all .h5 files in the target directory
    for target_file in target_path.glob('*.h5'):
        # Check if the unique identifier is present in the target filename
        if file_identifier in target_file.name:
            # If a match is found, return its full path as a string
            return str(target_file)
            
    # If the loop finishes without finding a match, return None
    return None


def get_atl08(atl08_file, gt):
    f = h5py.File(atl08_file, 'r') 
    
    # Read photon rate ATL03 data
    lon_ph = np.array(f[gt + '/heights/lon_ph'])
    lat_ph = np.array(f[gt + '/heights/lat_ph'])
    h_ph = np.array(f[gt + '/heights/h_ph'])
    quality_ph = np.array(f[gt + '/heights/quality_ph'])
    delta_time = np.array(f[gt + '/heights/delta_time'])
    
    # Read ATL03 segment rate at ATL03 photon rate
    solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/solar_elevation')
    alongtrack = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/segment_dist_x')
    alongtrack = alongtrack + np.array(f[gt + '/heights/dist_ph_along'])
    


if __name__ == "__main__":
    
    Define ATL03 File
    atl03_dir = '/Data/ICESat-2/REL006/florida_aoi/atl03'
    # atl03_name = 'ATL03_20200212094736_07280607_006_01.h5'
    atl03_file = os.path.join(atl03_dir, atl03_name)
    
    # Define ATL08 File
    atl08_dir = '/Data/ICESat-2/REL006/florida_aoi/atl08'
    # atl08_name = 'ATL08_20200212094736_07280607_006_01.h5'
    atl08_file = os.path.join(atl08_dir, atl08_name)
    
    # Define ATL24 file
    atl24_dir = '/Data/ICESat-2/REL006/florida_aoi/atl24'
    # atl24_name = 'ATL24_20200212094736_07280607_006_01_001_01.h5'
    atl24_file = os.path.join(atl24_dir, atl24_name)
    
    target_res = 30
    
    # Define Granule
    # gt = 'gt3r'
    
    atl24_list = os.listdir(atl24_dir)
    atl03_filelist = []
    atl24_filelist = []
    atl08_filelist = []
    
    for i in range(1,len(atl24_list)):
        atl03_file = find_corresponding_atl(atl24_list[i], atl03_dir)
        atl08_file = find_corresponding_atl(atl24_list[i], atl08_dir)
        atl03_filelist.append(atl03_file)
        atl08_filelist.append(atl08_file)
        atl24_filelist.append(os.path.join(atl24_dir,atl24_list[i]))
        
    gt_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']
    
    for i in range(0,len(atl24_filelist)):
        print(i)
        for gt in gt_list:
            try:
                atl03_file = atl03_filelist[i]
                atl08_file = atl08_filelist[i]
                atl24_file = atl24_filelist[i]
    
            
                file_out_name = f"{Path(atl03_file).stem}_{gt}"
                
                # Get photon rate DF
                df_ph = get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt)
                
                df_ph.loc[(df_ph['atl24_class'] == 40) & (df_ph['atl24_conf'] > 0.6),'atl24_class'] = 0
                
                
                
                # atl24_class[(df_ph['atl24_class'] == 40) & (df_ph['atl24_conf'] < 0.6)] = 0



                
                df_ph.atl08_class = df_ph.atl08_class.astype(int)
                # Aggregate photon rate DF to 10 m segment 
                df_seg = get_atl_at_seg(df_ph, res = target_res)
                
                # Write out as geopandas dataframe
                geometry_seg = [Point(xy) for xy in zip(df_seg.longitude, df_seg.latitude)]
                gdf_seg = gpd.GeoDataFrame(df_seg, geometry=geometry_seg, crs="EPSG:4326")
                
                # Save geopandas dataframe
                gdf_seg.to_file("is2_topobathy_" + gt + "_test.gpkg", layer='atl08atl24', driver="GPKG")
                
                
                # Read las extent
                extent_gpkg = '/home/ejg2736/dev/crossover_analysis/fl_west_Everglades_laz_extent1.gpkg'
                extent_gdf = gpd.read_file(extent_gpkg)
                
                # Trim Data
                df_ph = filter_df_by_extent(df_ph, extent_gdf.total_bounds)
                
                if len(df_ph) == 0:
                    print('No data in extent')
                    continue
                
                # Calculate norm height for topobathy
                df_ph['h_topobathy_norm'] = analysis.normalize_heights(df_ph, class_field = 'combined_class', 
                                  ground_class = [1,40], 
                                  ground_res = 5, 
                                  target_height = 'h_ph')
                
                # Calculate norm height for topobathy
                df_ph['h_toposurf_norm'] = analysis.normalize_heights(df_ph, class_field = 'combined_class', 
                                  ground_class = [1,41], 
                                  ground_res = 5, 
                                  target_height = 'h_ph')
                
                # Calculate norm height for topobathy
                df_ph['h_surf_norm'] = analysis.normalize_heights(df_ph, class_field = 'combined_class', 
                                  ground_class = [41], 
                                  ground_res = 5, 
                                  target_height = 'h_ph')
                
                # Calculate norm height for topobathy
                df_ph['h_te_norm'] = analysis.normalize_heights(df_ph, class_field = 'atl08_class', 
                                  ground_class = [1], 
                                  ground_res = 5, 
                                  target_height = 'h_ph')
                
                # Apply EGM2008
                geoid_file = '/dev/geoid/BundleAll/egm08_1.gtx'
                geoid_offset = get_geoid_height(np.array(df_ph.longitude), np.array(df_ph.latitude), geoid_file)
                df_ph['ortho_h'] = df_ph.h_ph - geoid_offset
                
                # Aggregate photon rate DF to 10 m segment 
                df_seg = get_atl_at_seg(df_ph, res = target_res)
                
                # Find best UTM zone
                utm_epsg = find_utm_zone_epsg(extent_gdf.iloc[0].lat_min,extent_gdf.iloc[0].lon_min)
                extent_gdf = extent_gdf.to_crs(utm_epsg) # Convert extent to UTM
                
                # Read ALS tile
                # If file already exists, skip processing
                als_outdir = '/Data/workspace/IS2/mangrove_fl/als'
                als_outfile = os.path.join(als_outdir, 'als_' + file_out_name + '.pqt')
                
                if os.path.exists(als_outfile):
                    als_swath = pd.read_parquet(als_outfile)
                    als_swath = als_swath.drop('key_id', axis=1)
                    
                else:
            
                    als_swath = create_als_swath(extent_gdf, df_seg)
                    
                    if len(als_swath) == 0:
                        print('No data ALS in extent')
                        continue
    
                    
                    # Calculate norm height for topobathy
                    als_swath['h_topobathy_norm'] = analysis.normalize_heights(als_swath, class_field = 'classification', 
                                      ground_class = [2,40], 
                                      ground_res = 1, 
                                      target_height = 'z')
                    
                    # Calculate norm height for topobathy
                    als_swath['h_norm'] = analysis.normalize_heights(als_swath, class_field = 'classification', 
                                      ground_class = [2, 41], 
                                      ground_res = 1, 
                                      target_height = 'z')
                                        
                    # Calculate Ellipsoid Height
                    geoid_file = '/dev/geoid/agisoft/us_noaa_g2012b.tif'
                    
                    transformer = Transformer.from_crs(int(utm_epsg[5:]),4326)
                    als_lat, als_lon  =transformer.transform(als_swath.x, als_swath.y)
                    geoid_offset = get_geoid_height(als_lon + 360, als_lat, geoid_file)
                    als_swath['ellip_h'] = als_swath.z + geoid_offset
                    als_swath['latitude'] = als_lat
                    als_swath['longitude'] = als_lon
                    
                    _,_,als_swath.ellip_h = convert_3d_nad83_to_wgs84(als_lon, als_lat, als_swath.ellip_h)
                    # test = test - 1.618700000000004
                    
                    
                    # Apply EGM2008
                    geoid_file = '/dev/geoid/BundleAll/egm08_1.gtx'
                    geoid_offset = get_geoid_height(als_lon, als_lat, geoid_file)
                    als_swath['ortho_h'] = als_swath.ellip_h - geoid_offset
                
                
                # als_swath.loc[(als_swath['classification'] == 1)  & (als_swath['h_norm'] > 0.1),'classification'] = 3
                als_swath.loc[(als_swath['classification'] == 1),'classification'] = 3

                als_seg = get_als_at_seg(als_swath, res = target_res, min_at = np.min(df_ph.alongtrack))
                
                #Write als_seg
                
                # Drop als_seg.alongtrack
                als_seg_modified = als_seg.drop(columns=['alongtrack'])
                
                als_seg_modified['als_comp_flag'].fillna('', inplace=True)
                # als_seg_modified['als_comp_flag'] = als_seg_modified['als_comp_flag'].astype(str)
                
                merged_df = pd.merge(df_seg, als_seg_modified, on='key_id', how='left')
                merged_df['als_comp_flag'].fillna('na', inplace=True)
                # Append information
                atl_info = get_attribute_info(atl03_file,gt)
                merged_df['atl03_file'] = os.path.basename(atl03_file)
                merged_df['gt'] = gt
                merged_df['atlas_beam_type'] = atl_info['atlas_beam_type']
                merged_df['atlas_spot_number'] = atl_info['atlas_spot_number']
                merged_df['year'] = atl_info['year']
                merged_df['doy'] = atl_info['doy']
                
                # Write out ATL_df
                alt_outdir = '/nfs_share/Data/workspace/IS2/mangrove_fl/atl'
                alt_outfile = os.path.join(alt_outdir, 'atl08atl24_' + file_out_name + '.pqt')
                df_ph.to_parquet(alt_outfile)
                
                # Write out ALS_df
                als_outdir = '/nfs_share/Data/workspace/IS2/mangrove_fl/als'
                als_outfile = os.path.join(als_outdir, 'als_' + file_out_name + '.pqt')
                als_swath.to_parquet(als_outfile)
                
                # Write out merged_df
                merged_outdir = '/nfs_share/Data/workspace/IS2/mangrove_fl/merged'
                merged_outfile = os.path.join(merged_outdir, 'merged' + str(target_res) + 'm_' + file_out_name + '.pqt')
                merged_df.to_parquet(merged_outfile)
                
                
                # Write out merged_csv
                mergedcsv_outdir = '/nfs_share/Data/workspace/IS2/mangrove_fl/merged_csv'
                merged_outfilecsv = os.path.join(merged_outdir, 'merged' + str(target_res) + 'm_' + file_out_name + '.csv')
                merged_df.to_csv(merged_outfilecsv)
                
                # Write out merged_gdf
                
                geometry = gpd.points_from_xy(merged_df['longitude'], merged_df['latitude'])
                merged_gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")
                
                mergedgdf_outdir = '/nfs_share/Data/workspace/IS2/mangrove_fl/merged_gdf'
                mergedgdf_outfile = os.path.join(mergedgdf_outdir, 'merged30m_' + file_out_name + '.gpkg')
                merged_gdf.to_file(mergedgdf_outfile,driver="GPKG")
            except:
                print('***')
                print(atl03_filelist[i])
                print(gt)
                print('Fail')
                print('***')


                
