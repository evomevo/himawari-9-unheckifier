import os
import warnings
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patheffects as PathEffects
from satpy import Scene
from pyresample import geometry
from pyproj import CRS
import dask

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

dask.config.set({
    "array.slicing.split_large_chunks": True,
    "scheduler": "threads"
})
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Uncomment the following lines if you obtained the script from the repo and want to run it.
# import subprocess
# subprocess.run(["python", "grab_raw_h9_data.py"])
# subprocess.run(["python", "grab_raw_goes_data.py"])
# subprocess.run(["python", "grab_raw_jpss_data.py"])
# print("Finished grabbing raw data. Initiating rendering...")

# Band choices
h9_band = 'B13'
goes_band = 'C13'

# Display parameters
vmin, vmax = 190, 310
cmap = plt.cm.bone_r

# Grid & resolution
gridline_spacing = 5 # degrees
resolution = 0.1 # degrees per pixel

# Full Plate Carrée bounds (do NOT change these!)
LON_MIN_FULL, LAT_MIN_FULL = -180, -90
LON_MAX_FULL, LAT_MAX_FULL = 180, 90

# Cropping region (adjust to preference)
lon_min_crop, lat_min_crop = -180, -90
lon_max_crop, lat_max_crop = 180, 90

# Overflow region for Himawari stitching (180° -> 225°E)
lon_overflow_start = 180.0
lon_overflow_end   = 250.0

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

# ─────────────────────────────────────────────────────────────────────────────
# 1) LOAD & RESAMPLE EVERY SATELLITE INTO FULL PLATE CARREE
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample_himawari(base_dir, band):
    """
    - Uses your exact snippet to find the latest "YYYYMMDD_HHMM" folder under base_dir
    - Loads all .DAT files for the requested band (e.g. 'B13')
    - Resamples to Plate Carrée (-180->+180) + does the 180->225°E overflow stitching.
    - Returns a (H_full, W_full) NumPy array of floats, with NaNs for invalid pixels.
    """
    # -------------------- your folder-finding snippet --------------------
    dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and len(d) == 13 and d[8] == '_'
    ]
    if not dirs:
        raise ValueError("No valid directories found in " + base_dir)

    latest_dir = max(dirs, key=lambda d: datetime.strptime(d, "%Y%m%d_%H%M"))
    raw_dir = os.path.join(base_dir, latest_dir)
    dat_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".DAT")]

    print(f"[Himawari-9] Found {len(dat_files)} .DAT files in {raw_dir}:")
    for f in sorted(dat_files):
        print("  ", f)
    # -----------------------------------------------------------------------

    # 2) create Satpy scene & load band
    scn = Scene(filenames=dat_files, reader="ahi_hsd")
    scn.load([band])

    # 3) Resample to Plate Carrée (-180->+180)
    area_def = geometry.AreaDefinition(
        'main_only',
        'PlateCarree (-180->+180 only)',
        'epsg:4326',
        crs.to_proj4(),
        width=W_full,
        height=H_full,
        area_extent=[LON_MIN_FULL, LAT_MIN_FULL, 180.0, LAT_MAX_FULL]
    )
    scn_resampled = scn.resample(area_def, resampler='nearest', radius_of_influence=50000)
    arr = scn_resampled[band].data.compute()

    # 4) Handle the overflow stitching (180° -> 225°E)
    lon, _ = scn_resampled[band].area.get_lonlats()
    arr[lon > 180.0] = np.nan

    # 5) Define the overflow area (-180 -> -135)
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
    return arr


def load_and_resample_goes(base_dir, goes_id, band):
    """
    - Finds latest YYYYDDD_HHMM folder under base_dir (len=12, '_' at idx 7).
    - Picks newest C13 file containing "_G{goes_id}_".
    - Loads it via Satpy, then:
        1) Resamples to PlateCarree (-180->+180).
        2) ALSO resamples to the “overflow” window [-180->-135] (i.e. 180->225°E).
        3) Stitches that small slice into the columns 0:W_B of the full array.
        4) Applies a cutoff at 110°W:
            - if goes_id == "19": keep only longitudes ≥ -110 (east of 110 W),
            mask out (NaN) everything west of -110.
            - if goes_id == "18": keep only longitudes ≤ -110 (west of 110 W),
            mask out (NaN) everything east of -110.
    - Returns a (H_full, W_full) array with NaNs where there's no data.
    """
    # 1) find latest GOES folder (YYYYDDD_HHMM, len=12, '_' at pos 7)
    dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and len(d) == 12 and d[7] == '_'
    ]
    if not dirs:
        raise ValueError(f"No valid directories found in {base_dir}")
    latest_dir = max(dirs, key=lambda d: datetime.strptime(d, "%Y%j_%H%M"))
    raw_dir = os.path.join(base_dir, latest_dir)
    print(f"[GOES-{goes_id}] Checking folder: {raw_dir}")

    # 2) pick newest C13 file with "_G{goes_id}_"
    goes_pattern = f"_G{goes_id}_"
    all_files = [f for f in os.listdir(raw_dir) if "C13" in f and goes_pattern in f]
    if not all_files:
        raise ValueError(f"No C13 files found for GOES-{goes_id} in {raw_dir}")
    latest_file = sorted(all_files)[-1]
    full_path = os.path.join(raw_dir, latest_file)
    print(f"[GOES-{goes_id}] Loading file: {latest_file}")

    # 3) load via Satpy's ABI reader
    scn = Scene(filenames=[full_path], reader="abi_l1b")
    scn.load([band])

    # 4a) Define Plate Carrée “main” area (-180->+180)
    area_def_A = geometry.AreaDefinition(
        "partA",
        "PlateCarree Main (-180->+180)",
        "epsg:4326",
        crs.to_proj4(),
        width=W_full,
        height=H_full,
        area_extent=[LON_MIN_FULL, LAT_MIN_FULL, LON_MAX_FULL, LAT_MAX_FULL]
    )
    scn_A = scn.resample(
        area_def_A,
        resampler="nearest",
        radius_of_influence=50000
    )
    arrA = scn_A[band].data.compute()  # shape = (H_full, W_full)

    # 4b) Define the “overflow” slice [180->225]E -> (-180->-135)
    lon_min_B = lon_overflow_start - 360.0  # = -180
    lon_max_B = lon_overflow_end   - 360.0  # = -135
    W_B = int((lon_max_B - lon_min_B) / resolution)  # same as Himawari's W_B
    H_B = H_full

    area_def_B = geometry.AreaDefinition(
        "partB",
        "PlateCarree Overflow (-180->-135)",
        "epsg:4326",
        crs.to_proj4(),
        width=W_B,
        height=H_B,
        area_extent=[lon_min_B, LAT_MIN_FULL, lon_max_B, LAT_MAX_FULL]
    )
    scn_B = scn.resample(
        area_def_B,
        resampler="nearest",
        radius_of_influence=50000
    )
    arrB = scn_B[band].data.compute()  # shape = (H_B, W_B)

    # 5) Stitch “Part B” (arrB) into the leftmost W_B columns of arrA
    arr_full = arrA.copy()
    arr_full[:, 0:W_B] = arrB

    # -------------------------------------------------------
    # 5.5) APPLY 110°W CUTOFF, **before** returning arr_full:
    #
    #   For GOES-19: keep lon ≥ -110°  (east of 110 W)
    #   For GOES-18: keep lon ≤ -110°  (west of 110 W)
    # -------------------------------------------------------

    # Build a 1D lon array at pixel CENTERS for the full Plate Carrée (W_full columns):
    #   column j corresponds to lon = LON_MIN_FULL + (j + 0.5)*resolution
    lon_vals = LON_MIN_FULL + (np.arange(W_full) + 0.5) * resolution

    if goes_id == "19":
        # keep only longitudes ≥ -110° (east of 110 W)
        lon_mask_1d = lon_vals >= -110.0
    else:  # goes_id == "18"
        # keep only longitudes ≤ -110° (west of 110 W)
        lon_mask_1d = lon_vals <= -110.0

    # Turn into 2D broadcast (H_full × W_full)
    lon_mask_2d = np.broadcast_to(lon_mask_1d, (H_full, W_full))

    # Zero‐out (→ NaN) any pixels where mask is False
    arr_full[~lon_mask_2d] = np.nan

    # 6) Finally, force NaN where invalid & cast to float32
    arr_full = np.where(np.isfinite(arr_full), arr_full, np.nan).astype(np.float32)
    return arr_full

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD ALL SATELLITES
# ─────────────────────────────────────────────────────────────────────────────

h9_arr = load_and_resample_himawari("himawari9_raw", h9_band)

g19_arr  = load_and_resample_goes("goes-19_raw", "19", goes_band)

g18_arr  = load_and_resample_goes("goes-18_raw", "18", goes_band)

# ─────────────────────────────────────────────────────────────────────────────
# 3) COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────

composite = np.where(
    np.isfinite(g19_arr), g19_arr,
    np.where(np.isfinite(h9_arr),
        h9_arr,
        g18_arr
    )).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4) CROP & PLOT
# ─────────────────────────────────────────────────────────────────────────────

lat_min_crop = max(lat_min_crop, LAT_MIN_FULL)
lat_max_crop = min(lat_max_crop, LAT_MAX_FULL)

def wrap_lon(lon):
    if lon > 180:
        return lon - 360
    if lon < -180:
        return lon + 360
    return lon

lon_min_wrapped = wrap_lon(lon_min_crop)
lon_max_wrapped = wrap_lon(lon_max_crop)
row_start = int((LAT_MAX_FULL - lat_max_crop) / resolution)
row_end   = int((LAT_MAX_FULL - lat_min_crop) / resolution)

if lon_min_crop >= -180 and lon_max_crop <= 180:
    col_start = int((lon_min_crop - LON_MIN_FULL) / resolution)
    col_end   = int((lon_max_crop - LON_MIN_FULL) / resolution)
    arr_crop  = composite[row_start:row_end, col_start:col_end]
    extent_crop = [lon_min_crop, lon_max_crop, lat_min_crop, lat_max_crop]
else:
    # wrap-around slicing
    lon1_min = lon_min_crop
    lon1_max = 180
    col1_start = int((lon1_min - LON_MIN_FULL) / resolution)
    col1_end   = int((lon1_max - LON_MIN_FULL) / resolution)
    part1 = composite[row_start:row_end, col1_start:col1_end]

    lon2_min = wrap_lon(max(lon_min_crop, 180))
    lon2_max = wrap_lon(lon_max_crop)
    col2_start = int((lon2_min - LON_MIN_FULL) / resolution)
    col2_end   = int((lon2_max - LON_MIN_FULL) / resolution)
    part2 = composite[row_start:row_end, col2_start:col2_end]

    arr_crop = np.hstack([part1, part2])
    extent_crop = [lon_min_crop, lon_max_crop, lat_min_crop, lat_max_crop]

height_pixels, width_pixels = arr_crop.shape

fig = plt.figure(
    frameon=False,
    figsize=(width_pixels / dpi, height_pixels / dpi),
    dpi=dpi
)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
ax.set_extent(extent_crop, ccrs.PlateCarree())

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
lon_ticks = np.arange(extent_crop[0], min(extent_crop[1], 180) + 1, gridline_spacing)
if extent_crop[1] > 180:
    wrapped_lo = wrap_lon(180 + gridline_spacing)
    wrapped_hi = wrap_lon(extent_crop[1])
    lon_ticks_over = np.arange(wrapped_lo, wrapped_hi + 1, gridline_spacing)
    lon_ticks = np.concatenate([lon_ticks, lon_ticks_over])

lat_ticks = np.arange(extent_crop[2], extent_crop[3] + 1, gridline_spacing)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    xlocs=lon_ticks,
    ylocs=lat_ticks,
    linewidth=1,
    color='white',
    linestyle='--'
)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.draw_labels = False

for lon in lon_ticks[::2]:
    for lat in lat_ticks[::2]:
        if int(lon) == 180:
            lon_label = "180°"
        elif int(lon) == 0:
            lon_label = "0°"
        else:
            lon_label = f"{abs(int(lon))}°{'E' if lon >= 0 else 'W'}"
        if int(lat) == 0:
            lat_label = "0°"
        else:
            lat_label = f"{abs(int(lat))}°{'N' if lat >= 0 else 'S'}"
        txt = ax.text(
            lon, lat,
            f"{lon_label}, {lat_label}",
            transform=ccrs.PlateCarree(),
            fontsize=10,
            color='white',
            ha='center',
            va='center',
            clip_on=True,
            fontweight='medium'
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

ax.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# 5) SAVE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

current_time = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
output_filename = f'composite_imagery_render_{current_time}.png'
fig.savefig(output_filename, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Successfully saved composited image as {output_filename}.")