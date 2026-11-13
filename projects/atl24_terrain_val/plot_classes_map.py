"""Photon classifications with a map: ATL08 (top left), ATL24 (bottom left), and
the ground track over USGS aerial imagery (right).

Same profile panels as plot_classes.py, plus a top-down view. On the map the
selected beam is drawn in yellow with along-track km marks that match the
profile x-axis; any other beams inside the map are thin white lines. When you
zoom or pan the profiles, the visible stretch is highlighted on the map.

Imagery: USGS National Map "USGSImageryOnly" (public domain), fetched for the
window's bounding box in NAD83 / UTM, so needs internet access.

Edit the settings below and run the file, or run it cell by cell in VS Code.
"""
# %% settings
import io
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from build_segments import SECONDS_PER_YEAR, dem_paths
from phoreal.config import load_config
from phoreal.geo.dem import DemStack, itrf_to_dem_frame

BEAM = 'gt1l'
LAT_RANGE = (38.73, 38.77)      # degrees north; keep the window small (~5 km), beams hold millions of photons
# LAT_RANGE = (38.77, 39.77)      # degrees north; keep the window small (~5 km), beams hold millions of photons
POINT_SIZE = 2
DEM_BIN_M = 2                   # along-track spacing of the 3DEP line (m)
MAP_BUFFER_M = 600              # imagery margin around the selected beam (m); raise it to see the other beams
MAP_MAX_PX = 1500               # longest side of the requested image (pixels)
IMAGERY_URL = 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export'

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
cfg = load_config(HERE / 'config.yaml')


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


work = resolve(cfg['work_dir'])

ATL08_CLASSES = {  # code: (label, colour); drawn in this order, so ground lands on top
    0: ('unclassified', '#9a9996'),
    2: ('vegetation', '#2ca02c'),
    3: ('vegetation', '#2ca02c'),
    1: ('ground', '#eb6834'),
}
ATL24_CLASSES = {
    0: ('unclassified', '#9a9996'),
    40: ('water', '#2a78d6'),
    41: ('water', '#2a78d6'),
    1: ('vegetation', '#2ca02c'),
    2: ('ground', '#eb6834'),
}


def read_window(gt, columns):
    return pd.read_parquet(work / 'photons' / f'photons_{gt}.parquet', columns=columns,
                           filters=[('latitude', '>=', min(LAT_RANGE)), ('latitude', '<=', max(LAT_RANGE))])


def ground_track(df, to_utm, bin_m=20):
    """Track centre line: median position of the signal photons per along-track bin."""
    s = df[(df['atl08_class'] > 0) | df['atl24_class'].isin([1, 2])]
    if s.empty:
        s = df
    e, n = to_utm.transform(s['longitude'].to_numpy(), s['latitude'].to_numpy())
    t = pd.DataFrame({'b': np.floor(s['alongtrack'].to_numpy() / bin_m), 'e': e, 'n': n,
                      'at': s['alongtrack'].to_numpy()})
    return t.groupby('b').median().sort_values('at')


# %% load one window of one beam
ph = read_window(BEAM, ['latitude', 'longitude', 'alongtrack', 'h_ph', 'delta_time', 'atl08_class', 'atl24_class'])
if ph.empty:
    raise SystemExit(f'no photons for {BEAM} in latitude {LAT_RANGE}')
x = ph['alongtrack'] / 1000
print(f'{len(ph):,} photons')

# %% 3DEP along the track (as ellipsoidal height, so it lines up with the photons)
line = None
paths = dem_paths(resolve(cfg['dem_dir']))
if paths:
    stack = DemStack(paths)
    epoch = 2018.0 + ph['delta_time'].mean() / SECONDS_PER_YEAR
    dx, dy, h_navd = itrf_to_dem_frame(ph['longitude'], ph['latitude'], ph['h_ph'], epoch, stack.crs)
    dem_ell = stack.sample(dx, dy)['elev'] + (ph['h_ph'].to_numpy() - h_navd)
    b = np.floor(ph['alongtrack'].to_numpy() / DEM_BIN_M).astype(np.int64)
    line = pd.DataFrame({'b': b, 'x': x.to_numpy(), 'dem': dem_ell}).groupby('b').median()
    line = line.reindex(np.arange(b.min(), b.max() + 1))   # empty bins -> NaN -> gaps in the line
    if line['dem'].isna().all():
        print('no 3DEP coverage in this window')
else:
    print('no staged 3DEP tiles: run stage_3dep.py to get the terrain line')

# %% ground tracks and imagery for the map
zone = int((ph['longitude'].median() + 180) // 6) + 1
utm_epsg = 26900 + zone                        # NAD83 / UTM, the 3DEP horizontal frame
to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}', always_xy=True)
cols = ['latitude', 'longitude', 'alongtrack', 'atl08_class', 'atl24_class']
tracks = {gt: ground_track(read_window(gt, cols) if gt != BEAM else ph, to_utm) for gt in cfg['beams']}
tracks = {gt: t for gt, t in tracks.items() if len(t)}

tb = tracks[BEAM]                              # map extent follows the selected beam
x0, x1 = tb['e'].min() - MAP_BUFFER_M, tb['e'].max() + MAP_BUFFER_M
y0, y1 = tb['n'].min() - MAP_BUFFER_M, tb['n'].max() + MAP_BUFFER_M
scale = MAP_MAX_PX / max(x1 - x0, y1 - y0)
params = {'bbox': f'{x0},{y0},{x1},{y1}', 'bboxSR': utm_epsg, 'imageSR': utm_epsg,
          'size': f'{round((x1 - x0) * scale)},{round((y1 - y0) * scale)}', 'format': 'jpg', 'f': 'image'}
try:
    with urllib.request.urlopen(f'{IMAGERY_URL}?{urllib.parse.urlencode(params)}', timeout=60) as r:
        img = plt.imread(io.BytesIO(r.read()), format='jpg')
except Exception as err:                      # still draw the tracks without imagery
    img = None
    print(f'imagery not available: {err}')

# %% plot
fig = plt.figure(figsize=(17, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.1])
ax08 = fig.add_subplot(gs[0, 0])
ax24 = fig.add_subplot(gs[1, 0], sharex=ax08, sharey=ax08)
axm = fig.add_subplot(gs[:, 1])

for ax, col, classes, title in ((ax08, 'atl08_class', ATL08_CLASSES, 'ATL08'),
                                (ax24, 'atl24_class', ATL24_CLASSES, 'ATL24')):
    seen = set()
    for code, (label, color) in classes.items():
        m = ph[col] == code
        if not m.any():
            continue
        ax.scatter(x[m], ph['h_ph'][m], s=POINT_SIZE, c=color, linewidths=0, rasterized=True,
                   label=None if label in seen else label)
        seen.add(label)
    if line is not None:
        ax.plot(line['x'], line['dem'], color='black', linewidth=1.2, label='3DEP DTM')
    other = ~ph[col].isin(list(classes))
    if other.any():
        print(f'{title}: {other.sum():,} photons with other codes not drawn: {sorted(ph[col][other].unique())}')
    ax.set_title(f'{title} photon classes, {BEAM}', loc='left')
    ax.set_ylabel('Ellipsoidal height (m)')
    ax.grid(color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', markerscale=4, frameon=False)
ax08.tick_params(labelbottom=False)
ax24.set_xlabel('Along-track distance (km)')

sig = ph['h_ph'][(ph['atl08_class'] > 0) | ph['atl24_class'].isin([1, 2])]
if len(sig):
    lo, hi = sig.quantile([0.001, 0.999])
    pad = max(10.0, 0.1 * (hi - lo))
    ax08.set_ylim(lo - pad, hi + pad)

# map: imagery, all beams, the selected beam with km marks
if img is not None:
    axm.imshow(img, extent=(x0, x1, y0, y1), origin='upper', interpolation='bilinear')
others = [gt for gt in tracks if gt != BEAM]
for i, gt in enumerate(others):                # other beams that fall inside the map
    t = tracks[gt]
    axm.plot(t['e'], t['n'], color='white', linewidth=0.8, alpha=0.8,
             label='other beams' if i == 0 else None)
axm.plot(tb['e'], tb['n'], color='#ffd400', linewidth=1.8, label=BEAM)
km = np.arange(np.ceil(tb['at'].min() / 1000), np.floor(tb['at'].max() / 1000) + 1)
ke, kn = np.interp(km * 1000, tb['at'], tb['e']), np.interp(km * 1000, tb['at'], tb['n'])
axm.scatter(ke, kn, s=18, color='#ffd400', edgecolors='black', linewidths=0.6, zorder=3)
for k, e, n in zip(km, ke, kn):
    axm.annotate(f'{k:.0f}', (e, n), xytext=(6, 0), textcoords='offset points', va='center',
                 fontsize=8, color='white', fontweight='bold')
(view,) = axm.plot([], [], color='#00e5ff', linewidth=4, alpha=0.7, label='profile view')
axm.set_xlim(x0, x1)
axm.set_ylim(y0, y1)
axm.set_aspect('equal')
axm.set_title(f'{BEAM} over USGS imagery\nUTM {zone}N, marks = along-track km', loc='left', fontsize=10)
axm.set_xlabel('Easting (m)')
axm.set_ylabel('Northing (m)')
axm.ticklabel_format(style='plain', useOffset=False)
axm.tick_params(axis='x', labelrotation=30)
axm.legend(loc='lower right', fontsize=8, framealpha=0.8)


def show_view(ax):
    """Highlight on the map the stretch of the beam visible in the profile panels."""
    a, b = (v * 1000 for v in ax.get_xlim())
    m = (tb['at'] >= a) & (tb['at'] <= b)
    full = m.all()
    view.set_data(tb['e'][m] if not full else [], tb['n'][m] if not full else [])
    fig.canvas.draw_idle()


ax08.callbacks.connect('xlim_changed', show_view)
plt.show()
