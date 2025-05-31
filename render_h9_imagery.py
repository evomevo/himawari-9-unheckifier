import numpy as np
import os, warnings, dask
from satpy import Scene
from pyresample import geometry
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as PathEffects
from pyproj import CRS
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime

dask.config.set({
    "array.slicing.split_large_chunks": True,
    "scheduler": "threads"
})

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define area with target pixels:
band = 'B13'
vmin, vmax = 190, 310
lon_min, lat_min = 95,  0  # 100°E, Equator
lon_max, lat_max = 180, 60  # 180°E, 60°N

# Set the desired resolution in degrees per pixel
resolution = 0.02 # degrees per pixel

# Calculate width and height in pixels based on the geographical span:
W = int((lon_max - lon_min) / resolution)
H = int((lat_max - lat_min) / resolution)
crs = CRS.from_string("EPSG:4326") # Plate Carrée = WGS84 with lat/lon degrees

font_paths = [f.fname for f in fm.fontManager.ttflist if 'Montserrat' in f.name]
if font_paths:
    montserrat_path = font_paths[0] # Use the first match
    plt.rcParams['font.family'] = fm.FontProperties(fname=montserrat_path).get_name()
else:
    raise ValueError("Montserrat not found. Make sure it's installed on your system.")

area_def = geometry.AreaDefinition(
    'global_eq',                    # ID
    'Global Plate Carree',          # name
    'epsg:4326',                    # projection (string ID)
    crs.to_proj4(),
    width=W, height=H,
    area_extent=[lon_min, lat_min, lon_max, lat_max]
)

# Load and resample Himawari-9 data:
base_dir = "himawari9_raw"
# List only valid subdirectories using the expected naming convention 'YYYYMMDD_HHMM'
dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and len(d) == 13 and d[8] == '_']
if not dirs:
    raise ValueError("No valid directories found in " + base_dir)

# Parse the directory names into datetime objects and find the most recent one
latest_dir = max(dirs, key=lambda d: datetime.strptime(d, "%Y%m%d_%H%M"))
raw_dir = os.path.join(base_dir, latest_dir)
dat_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if f.endswith(".DAT")]
print(latest_dir, raw_dir, dat_files)

scn = Scene(filenames=dat_files, reader="ahi_hsd")
scn.load([band])
scn_eq = scn.resample(area_def, resampler='nearest', radius_of_influence=50000)
ir = scn_eq[band].data

# Plot to a pixel-perfect canvas. No axes, no padding, no margins:
dpi = 100  # so that W = figsize[0]*dpi -> figsize[0] = W/dpi
fig = plt.figure(frameon=False, figsize=(W/dpi, H/dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

# Mask out invalid (off-limb) pixels so they’re transparent
ir_masked = np.ma.masked_invalid(ir)
cmap = plt.cm.bone_r # reversed to put colder temps as brighter

# Show your image (still use imshow but with proper extent):
ax.imshow(ir_masked, origin='upper',
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        interpolation='nearest')

# Add coast and grid lines:
ax.coastlines(resolution='10m', linewidth=2, color='white')

gridline_spacing = 5 # degrees
lon_ticks = np.arange(lon_min, lon_max + 1, gridline_spacing)
lat_ticks = np.arange(lat_min, lat_max + 1, gridline_spacing)

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

def format_coord(lon, lat):
    def format_single(value, positive_suffix, negative_suffix):
        suffix = positive_suffix if value >= 0 else negative_suffix
        return f"{abs(int(value))}°{suffix}"

    lon_str = format_single(lon, "E", "W")
    lat_str = format_single(lat, "N", "S")
    return f"{lon_str}, {lat_str}"

for lon in lon_ticks[::2]:
    for lat in lat_ticks[::2]:
        label = format_coord(lon, lat)
        txt = ax.text(lon, lat,
                      label,
                      transform=ccrs.PlateCarree(),
                      fontsize=10,
                      color='white',
                      ha='center', va='center',
                      clip_on=True,
                      fontweight='medium')

        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground='black')
        ])

ax.axis('off')

# Save
fileName = f'himawari-9_{band}_eq_{latest_dir}.png'
fig.savefig(f'{fileName}', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"Ding! Saved {fileName} with size {W}x{H} pixels.")