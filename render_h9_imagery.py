import os
import warnings
import dask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as PathEffects
from satpy import Scene
from pyresample import geometry
from pyproj import CRS
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

dask.config.set({
    "array.slicing.split_large_chunks": True,
    "scheduler": "threads"
})
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Uncomment the following lines if you obtained the script from the repo and want to run it.
# subprocess.run(["python", "grab_raw_h9_data.py"])
# print("grab_raw_h9_data.py has finished executing. Executing render_h9_imagery.py...")

# Band options are available in the repo README.
band = 'B13'
vmin, vmax = 190, 310

# Gridline & resolution
gridline_spacing = 5 # degrees
resolution = 0.07 # degrees per pixel

# Full Plate Carrée bounds (we will split this into two parts)
lon_min_full, lat_min_full = -180, -90
lon_max_full, lat_max_full =  180,  90

# The “overflow” region is 180->225 E (which is -180 -> -135 in -180..180)
# So Part A will cover [ -180, -90,  180,  90 ]
#    Part B will cover [ -180, -90, -135,  90 ]
lon_overflow_start = 180.0 # 180° E
lon_overflow_end   = 225.0 # 225° E  ->  in -180..180 that is -135° E

# Calculate W, H for the full Plate Carrée (Part A)
W_full = int((lon_max_full - lon_min_full) / resolution)
H_full = int((lat_max_full - lat_min_full) / resolution)

# CRS
crs = CRS.from_string("EPSG:4326") # Plate Carrée = WGS84 lon/lat

# Montserrat font
font_paths = [f.fname for f in fm.fontManager.ttflist if 'Montserrat' in f.name]
if font_paths:
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_paths[0]).get_name()
else:
    raise ValueError("Montserrat not found. Make sure it's installed on your system.")

# ─────────────────────────────────────────────────────────────────────────────
# 1) FIND THE LATEST RAW .DAT FOLDER
# ─────────────────────────────────────────────────────────────────────────────

base_dir = "himawari9_raw"
dirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and len(d) == 13 and d[8] == '_'
]
if not dirs:
    raise ValueError("No valid directories found in " + base_dir)

latest_dir = max(dirs, key=lambda d: datetime.strptime(d, "%Y%m%d_%H%M"))
raw_dir = os.path.join(base_dir, latest_dir)
dat_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".DAT")]

print(f"Found {len(dat_files)} .DAT files in {raw_dir}:")
for f in sorted(dat_files):
    print("  ", f)

# ─────────────────────────────────────────────────────────────────────────────
# 2) SET UP SATPY SCENE
# ─────────────────────────────────────────────────────────────────────────────

scn = Scene(filenames=dat_files, reader="ahi_hsd")
scn.load([band])

# ─────────────────────────────────────────────────────────────────────────────
# 3) PART A: RESAMPLE FOR [-180 -> +180] (the “main” Plate Carrée grid)
# ─────────────────────────────────────────────────────────────────────────────

area_def_A = geometry.AreaDefinition(
    'partA',
    'PlateCarree Main (-180->+180)',
    'epsg:4326',
    crs.to_proj4(),
    width=W_full,
    height=H_full,
    area_extent=[lon_min_full, lat_min_full, lon_max_full, lat_max_full]
)

scn_A = scn.resample(area_def_A, resampler='nearest', radius_of_influence=50000)
irA = scn_A[band].data.compute() # NumPy array of shape (H_full, W_full)

# ─────────────────────────────────────────────────────────────────────────────
# 4) PART B: RESAMPLE FOR [-180 -> -135] which corresponds to [180 -> 225] E overflow
# ─────────────────────────────────────────────────────────────────────────────

# Convert [180, 225] E -> [-180, -135] in -180..180 notation:
lon_min_B = lon_overflow_start - 360.0 # 180 - 360 = -180
lon_max_B = lon_overflow_end   - 360.0 # 225 - 360 = -135

# How many columns is that at the same resolution?
# (-135 - (-180)) / 0.03 = 35 / 0.03 ≈ 1166.66 -> ceil to 1167 or floor to 1166
W_B = int((lon_max_B - lon_min_B) / resolution)
H_B = H_full # same vertical span as Part A

area_def_B = geometry.AreaDefinition(
    'partB',
    'PlateCarrée Overflow (-180 -> -135)',
    'epsg:4326',
    crs.to_proj4(),
    width=W_B,
    height=H_B,
    area_extent=[lon_min_B, lat_min_full, lon_max_B, lat_max_full]
)

scn_B = scn.resample(area_def_B, resampler='nearest', radius_of_influence=50000)
irB = scn_B[band].data.compute() # NumPy array of shape (H_B, W_B)

# ─────────────────────────────────────────────────────────────────────────────
# 5) STITCH PART B INTO PART A
# ─────────────────────────────────────────────────────────────────────────────

# Make a final array = Part A (will modify leftmost columns)
ir_final = irA.copy()

# Part B's data should go into the leftmost W_B columns of Part A (-180 -> -135)
# In our PlateCarree grid, column index 0 = lon = -180
# So we overwrite [ :, 0:W_B ] with irB
ir_final[:, 0:W_B] = irB

# ─────────────────────────────────────────────────────────────────────────────
# 6) PLOT THE FINAL STITCHED IMAGE
# ─────────────────────────────────────────────────────────────────────────────

dpi = 100
fig = plt.figure(frameon=False, figsize=(W_full / dpi, H_full / dpi), dpi=dpi)

ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
ax.set_extent([lon_min_full, lon_max_full, lat_min_full, lat_max_full],
              ccrs.PlateCarree())

# Mask the NaNs (off-limb) to transparent
ir_plot = np.ma.masked_invalid(ir_final)
cmap = plt.cm.bone_r # Or choose any other colormap as mentioned in the repo

ax.imshow(
    ir_plot,
    origin='upper',
    extent=[lon_min_full, lon_max_full, lat_min_full, lat_max_full],
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree(),
    interpolation='nearest'
)

# Add coastlines and grid
ax.coastlines(resolution='10m', linewidth=2, color='white')

lon_ticks = np.arange(lon_min_full, lon_max_full + 1, gridline_spacing)
lat_ticks = np.arange(lat_min_full, lat_max_full + 1, gridline_spacing)

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

# Draw labels at intersections
def format_coord(lon, lat):
    def fs(val, pos_suf, neg_suf):
        suf = pos_suf if val >= 0 else neg_suf
        return f"{abs(int(val))}°{suf}"
    return f"{fs(lon,'E','W')}, {fs(lat,'N','S')}"

for lon in lon_ticks[::2]:
    for lat in lat_ticks[::2]:
        txt = ax.text(
            lon, lat,
            format_coord(lon, lat),
            transform=ccrs.PlateCarree(),
            fontsize=10,
            color='white',
            ha='center', va='center',
            clip_on=True,
            fontweight='medium'
        )
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground='black')
        ])

ax.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# 7) SAVE THE FINAL IMAGE
# ─────────────────────────────────────────────────────────────────────────────

fileName = f'himawari-9_{band}_eq_{latest_dir}.png'
fig.savefig(fileName, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Ding! Saved {fileName} with size {W_full}×{H_full} pixels.")
