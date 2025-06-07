import os, warnings, time
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
from satpy import Scene
from pyresample import geometry
from pyproj import CRS
import dask, matplotlib

# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------
matplotlib.use('Agg')

dask.config.set({
    "array.slicing.split_large_chunks": True,
    "scheduler": "threads"
})
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Uncomment the following lines if you obtained the script from the repo and want to run it.
# import subprocess
# subprocess.run(["python", "grab_raW_himawari_9_data.py"])
# subprocess.run(["python", "grab_raw_goes-18-19_data.py"])
# subprocess.run(["python", "grab_raw_jpss_data.py"])
# print("Finished grabbing raw data. Initiating rendering...")

# Start time
start = time.time()

# Band choices
h9_band = 'B13'
goes_band = 'C13'

# Grid & resolution
gridline_spacing = 5 # degrees
resolution = 0.05 # degrees per pixel

# Full Plate Carrée bounds (do NOT change these!)
LON_MIN_FULL, LAT_MIN_FULL = -180, -90
LON_MAX_FULL, LAT_MAX_FULL = 180, 90

# Cropping region (adjust to preference)
lon_min_crop, lat_min_crop = -180, -90
lon_max_crop, lat_max_crop = 180, 90

# Depending on if there are position changes or replacements, adjust these.
satellite_longitudes = [
    140.7, # Himawari-9 @ 140.7°E
    -75.2, # GOES-19 @ 75.2°W
    -136.9 # GOES-18 @ 136.9°W
]

# Display parameters
def ott():
    newcmp = LinearSegmentedColormap.from_list("ott", [
        (0/170, "#000000"),
        (15/170, "#000000"),
        (80/170, "#bfbfbf"),
        (80/170, "#00bfff"),
        (90/170, "#000080"),
        (100/170, "#00FF00"),
        (110/170, "#FFFF00"),
        (120/170, "#FF0000"),
        (130/170, "#000000"),
        (140/170, "#FFFFFF"),
        (140/170, "#ff80bf"),
        (150/170, "#800080"),
        (150/170, "#ffff00"),
        (160/170, "#000000"),
        (170/170, "#000000")
    ])
    
    vmax = 60 + 273.15
    vmin = -110 + 273.15

    return newcmp.reversed(), vmax, vmin

def dvorak():
    newcmp = LinearSegmentedColormap.from_list("", [
        (0/120, "#000000"),
        (21/120, "#fafafa"),
        (21/120, "#3a3a3a"),
        (60/120, "#d2d2d2"),
        (60/120, "#5b5b5b"),
        (71/120, "#5b5b5b"),
        (71/120, "#9a9a9a"),
        (83/120, "#9a9a9a"),
        (83/120, "#b7b7b7"),
        (93/120, "#b7b7b7"),
        (93/120, "#000000"),
        (99/120, "#000000"),
        (99/120, "#f9f9f9"),
        (105/120, "#f9f9f9"),
        (105/120, "#9e9e9e"),
        (110/120, "#9e9e9e"),
        (110/120, "#424242"),
        (120/120, "#424242")
    ])

    vmax = 30 + 273.15
    vmin = -90 + 273.15

    return newcmp.reversed(), vmax, vmin

cmap, vmax, vmin = ott()

# Figure DPI
dpi = 100

# Montserrat font (throws an error if not found)
font_paths = [f.fname for f in fm.fontManager.ttflist if 'Montserrat' in f.name]
if font_paths:
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_paths[0]).get_name()
else:
    raise ValueError("Montserrat not found. Make sure it's installed on your system.")

# CRS for Plate Carrée
crs = CRS.from_string("EPSG:4326")

# Precompute full image dimensions for Plate Carrée
W_full = int((LON_MAX_FULL - LON_MIN_FULL) / resolution)
H_full = int((LAT_MAX_FULL - LAT_MIN_FULL) / resolution)
MAIN_AREA = geometry.AreaDefinition(
    "full_disk", "PlateCarree Full",
    "epsg:4326", crs.to_proj4(),
    width=W_full, height=H_full,
    area_extent=[LON_MIN_FULL, LAT_MIN_FULL, LON_MAX_FULL, LAT_MAX_FULL]
)
# Himawari overflow
W_himawari_9 = int((225.0 - 180.0) / resolution)
OVERFLOW_HIMAWARI_9 = geometry.AreaDefinition(
    "HIM9_over", "Overflow H9",
    "epsg:4326", crs.to_proj4(),
    width=W_himawari_9, height=H_full,
    area_extent=[-180, LAT_MIN_FULL, -135, LAT_MAX_FULL]
)

# GOES overflow
W_goes = int((260.0 - 180.0) / resolution)
OVERFLOW_GOES = geometry.AreaDefinition(
    "GOES_over", "Overflow GOES",
    "epsg:4326", crs.to_proj4(),
    width=W_goes, height=H_full,
    area_extent=[-180, LAT_MIN_FULL, -100, LAT_MAX_FULL]
)

W_crop = int((lon_max_crop - lon_min_crop)/resolution)
H_crop = int((lat_max_crop - lat_min_crop)/resolution)
CROP_AREA = geometry.AreaDefinition(
    "crop", "User Crop",
    "epsg:4326", crs.to_proj4(),
    width=W_crop, height=H_crop,
    area_extent=[lon_min_crop, lat_min_crop, lon_max_crop, lat_max_crop]
)

# -----------------------------------------------------------------------------
# 1) LOAD & RESAMPLE EVERY SATELLITE INTO FULL PLATE CARREE
# -----------------------------------------------------------------------------
def get_latest_directory(base_dir, name_length, underscore_idx, dt_format):
    dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and len(d) == name_length and d[underscore_idx] == '_'
    ]
    if not dirs:
        raise ValueError("No valid directories found in " + base_dir)
    latest_dir = max(dirs, key=lambda d: datetime.strptime(d, dt_format))
    return os.path.join(base_dir, latest_dir)

def resample_and_stitch(scn, band, main_area, overflow_area, overflow_width, radius=50000):
    scn_main = scn.resample(main_area, resampler="nearest", radius_of_influence=radius)
    arr_main = scn_main[band].data.compute()
    
    scn_of = scn.resample(overflow_area, resampler="nearest", radius_of_influence=radius)
    arr_overflow = scn_of[band].data.compute()
    
    arr_full = arr_main.copy()
    arr_full[:, :overflow_width] = arr_overflow
    return np.where(np.isfinite(arr_full), arr_full, np.nan).astype(np.float32)

def load_and_resample_himawari(base_dir):
    # 1) Find latest raw folder using YYYYMMDD_HHMM (length==13, '_' at idx 8)
    raw_dir = get_latest_directory(base_dir, name_length=13, underscore_idx=8, dt_format="%Y%m%d_%H%M")
    dat_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".DAT")]
    print(f"[Himawari-9] Found {len(dat_files)} .DAT files in {raw_dir}")

    # 2) Load scene via Satpy
    scn = Scene(filenames=dat_files, reader="ahi_hsd")
    scn.load([h9_band])

    # 3) Resample and stitch overflow into main image
    return resample_and_stitch(scn, h9_band, MAIN_AREA, OVERFLOW_HIMAWARI_9, W_himawari_9)

def load_and_resample_goes(base_dir, goes_id):
    # 1) Find latest raw folder using YYYYDDD_HHMM (length==12, '_' at idx 7)
    raw_dir = get_latest_directory(base_dir, name_length=12, underscore_idx=7, dt_format="%Y%j_%H%M")
    print(f"[GOES-{goes_id}] Checking folder: {raw_dir}")

    # 2) Pick the newest file containing the band and goestag
    goes_pattern = f"_G{goes_id}_"
    files = [f for f in os.listdir(raw_dir) if goes_band in f and goes_pattern in f]
    if not files:
        raise ValueError(f"No {goes_band} files found for GOES-{goes_id} in {raw_dir}")
    latest_file = sorted(files)[-1]
    full_path = os.path.join(raw_dir, latest_file)
    print(f"[GOES-{goes_id}] Loading file: {latest_file}")

    # 3) Load scene via Satpy's ABI reader
    scn = Scene(filenames=[full_path], reader="abi_l1b")
    scn.load([goes_band])

    # 4) Resample and stitch overflow into the main image
    return resample_and_stitch(scn, goes_band, MAIN_AREA, OVERFLOW_GOES, W_goes)

# -----------------------------------------------------------------------------
# 2) LOAD ALL SATELLITES
# -----------------------------------------------------------------------------
h9_arr = load_and_resample_himawari("himawari-9_raw")

g19_arr  = load_and_resample_goes("goes-19_raw", "19")

g18_arr  = load_and_resample_goes("goes-18_raw", "18")

def get_satellite_cutoffs(lon_vals, satellite_longitudes):
    """
    lon_vals: 1D array of longitudes in degrees (e.g. -180 to +180 or 0-360)
    satellite_longitudes: list of each sat's sub-satellite longitude (in same convention)
    Returns: masks of shape (N_sats, len(lon_vals)), where each row is True inside that sat's domain.
    """

    # 1) Normalize both lon_vals & subsat positions into 0-360°
    L = (lon_vals + 360) % 360
    S = (np.array(satellite_longitudes) + 360) % 360
    S_sorted = np.sort(S)

    # 2) Sort sats (and keep original order)
    order = np.argsort(S)

    # 3) Compute midpoints on the circle
    mids = (S_sorted + np.roll(S_sorted, -1)) / 2
    mids[-1] = ((S_sorted[-1] + (S_sorted[0] + 360)) / 2) % 360

    # 4) Build masks between each pair of mids (with wrap if needed)
    lows  = np.roll(mids, 1)
    highs = mids
    masks_sorted = np.zeros((len(S_sorted), L.size), dtype=bool)
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        if lo < hi:
            masks_sorted[i] = (L >= lo) & (L < hi)
        else:
            masks_sorted[i] = (L >= lo) | (L < hi)

    # 5) Reorder back to original satellite list order
    masks = masks_sorted[np.argsort(order)]
    return masks

lon_vals = LON_MIN_FULL + (np.arange(W_full) + 0.5) * resolution

sat_masks = get_satellite_cutoffs(lon_vals, satellite_longitudes)

h9_arr[:,  ~sat_masks[0]] = np.nan # Himawari
g19_arr[:, ~sat_masks[1]] = np.nan # GOES-19
g18_arr[:, ~sat_masks[2]] = np.nan # GOES-18

# -----------------------------------------------------------------------------
# 3) COMPOSITE, CROP, AND PLOT
# -----------------------------------------------------------------------------
composite = np.where(
    np.isfinite(g19_arr), g19_arr,
    np.where(np.isfinite(h9_arr), h9_arr, g18_arr)
).astype(np.float32)

lat_min_crop = max(lat_min_crop, LAT_MIN_FULL)
lat_max_crop = min(lat_max_crop, LAT_MAX_FULL)

def wrap_lon(lon):
    return ((lon + 180) % 360) - 180

# Compute row indices (latitude)
row_start = int((LAT_MAX_FULL - lat_max_crop) / resolution)
row_end   = int((LAT_MAX_FULL - lat_min_crop) / resolution)

# Compute column indices (longitude)
if lon_min_crop >= -180 and lon_max_crop <= 180:
    col_start = int((lon_min_crop - LON_MIN_FULL) / resolution)
    col_end   = int((lon_max_crop - LON_MIN_FULL) / resolution)
    arr_crop = composite[row_start:row_end, col_start:col_end]
else:
    # First section: from lon_min_crop up to 180
    col1_start = int((lon_min_crop - LON_MIN_FULL) / resolution)
    col1_end   = int((180 - LON_MIN_FULL) / resolution)
    part1 = composite[row_start:row_end, col1_start:col1_end]
    # Second section: from -180 up to wrapped lon_max_crop
    col2_start = int((-180 - LON_MIN_FULL) / resolution)
    col2_end   = int((wrap_lon(lon_max_crop) - LON_MIN_FULL) / resolution)
    part2 = composite[row_start:row_end, col2_start:col2_end]
    arr_crop = np.hstack([part1, part2])

extent_crop = [lon_min_crop, lon_max_crop, lat_min_crop, lat_max_crop]

height_pixels, width_pixels = arr_crop.shape
fig = plt.figure(figsize=(width_pixels / dpi, height_pixels / dpi), dpi=dpi, frameon=False)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
ax.set_extent(extent_crop, crs=ccrs.PlateCarree())

ax.imshow(
    np.ma.masked_invalid(arr_crop),
    origin='upper',
    extent=extent_crop,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree(),
    interpolation='nearest'
)
ax.coastlines(resolution='10m', linewidth=2, color='white')

# Set grid and tick labels
lon_ticks = np.arange(extent_crop[0], extent_crop[1] + gridline_spacing, gridline_spacing)
lat_ticks = np.arange(extent_crop[2], extent_crop[3] + gridline_spacing, gridline_spacing)

gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=lon_ticks, ylocs=lat_ticks,
                  linewidth=1, color='white')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.draw_labels = False

def format_label(val, is_lon=True):
    val_int = int(np.round(val))
    if is_lon:
        if val_int in (-180, 180, 0):
            return f"{val_int}°"
        return f"{abs(val_int)}°{'E' if val >= 0 else 'W'}"
    else:
        if val_int == 0:
            return "0°"
        return f"{abs(val_int)}°{'N' if val >= 0 else 'S'}"

# Add labels at grid intersections (every other tick)
for lon in lon_ticks[::2]:
    for lat in lat_ticks[::2]:
        ax.text(lon, lat, f"{format_label(lon, True)}, {format_label(lat, False)}",
                transform=ccrs.PlateCarree(), fontsize=10, color='white', fontweight='medium',
                ha='center', va='center', clip_on=True,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

ax.axis('off')

# -----------------------------------------------------------------------------
# 4) SAVE IMAGE
# -----------------------------------------------------------------------------
current_time = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
output_filename = f'composite_imagery_render_{current_time}.png'
fig.savefig(output_filename, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig)

end = time.time()
print(f"Successfully saved composited image as {output_filename} in {end - start} seconds.")