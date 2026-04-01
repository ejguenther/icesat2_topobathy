import h5py
import os
from pathlib import Path

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