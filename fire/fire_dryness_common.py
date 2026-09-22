"""Shared helpers for the fire-event dryness scripts (fire/plot_fire_event_maps.py,
fire/plot_fire_overview.py, fire/check_fuel_dryness_baseline.py).

Not a script itself -- import it from the same directory (`fire/`).

Colour design follows this repo's dataviz conventions: one-hue blue ramp for a sequential
(absolute-value) map, diverging red<->blue with a neutral midpoint for a dryness-rank map
(dry = red). The blue ramp and the red ramp's mid-step (`#e34948`) come from the project's
validated palette; the lighter/darker red steps are derived here since that palette does not
document a full red ramp.
"""
from __future__ import annotations

import warnings

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN reductions over sea cells

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7975"
AXIS, GRID = "#d3d2cd", "#ecebe7"
BLUE = "#2a78d6"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
_DRY = ["#8e1f22", "#e34948", "#ec8783", "#f3c3bd"]  # dark -> light; #e34948 is the palette's red
_WET = ["#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"]  # light -> dark; all palette blue steps
DIVERGING_DRY_WET = LinearSegmentedColormap.from_list("dry_wet", _DRY + ["#f0efec"] + _WET)

# Reference cities shown on the first panel of each figure, purely for orientation -- not data.
REFERENCE_CITIES = {
    "Barcelona": (2.17, 41.39), "Zaragoza": (-0.88, 41.65), "Toulouse": (1.44, 43.60),
    "Marseille": (5.37, 43.30), "Bordeaux": (-0.58, 44.84), "Perpignan": (2.90, 42.70),
}


def cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell-edge coordinates from a regular grid's cell-center coordinates."""
    mid = (centers[1:] + centers[:-1]) / 2
    return np.concatenate([[centers[0] - (mid[0] - centers[0])], mid, [centers[-1] + (centers[-1] - mid[-1])]])


def day_index(dates: np.ndarray, year: int, month_day: str) -> int | None:
    """Index of `<year>-<month_day>` in a sorted `datetime64[D]` array, or None if absent
    (e.g. 29 February in a non-leap year, or a date outside the cached range)."""
    try:
        t = np.datetime64(f"{year}-{month_day}")
    except ValueError:
        return None
    i = np.searchsorted(dates, t)
    return int(i) if i < len(dates) and dates[i] == t else None


def nearest_land_cell(lat: np.ndarray, lon: np.ndarray, land: np.ndarray, lat0: float, lon0: float) -> tuple[int, int]:
    d2 = (lon[None, :] - lon0) ** 2 + (lat[:, None] - lat0) ** 2
    d2 = np.where(land, d2, np.inf)
    j, i = np.unravel_index(np.argmin(d2), d2.shape)
    return int(j), int(i)


def wetter_share(event_value: np.ndarray, other_years: np.ndarray, eps: float) -> np.ndarray:
    """Per-cell share of `other_years` that were WETTER than `event_value` (ties count half);
    1 = the driest that date has been across every year in `other_years`."""
    n = np.isfinite(other_years).sum(0).astype("f8")
    n[n == 0] = np.nan
    wetter = (other_years > event_value[None] + eps).sum(0)
    tied = (np.abs(other_years - event_value[None]) <= eps).sum(0)
    return (wetter + 0.5 * tied) / n


def rank_text(value: float, other_years_at_cell: np.ndarray, fmt, eps: float) -> str:
    c = other_years_at_cell[np.isfinite(other_years_at_cell)]
    drier = int((c > value + eps).sum())
    tied = int((np.abs(c - value) <= eps).sum())
    text = f"fire cell {fmt(value)} (other years {fmt(c.mean())}); drier than {drier} of {len(c)} years"
    return text + (f", {tied} tied" if tied else "")


_geometry_cache: dict[str, list] = {}


def domain_geometries(extent: list[float]) -> dict[str, list]:
    """10 m Natural Earth coastline/border geometries clipped to `extent` (+1 deg margin),
    downloaded via cartopy's own cache on first use -- no machine-specific path assumed."""
    key = ",".join(f"{v:.3f}" for v in extent)
    if key not in _geometry_cache:
        from shapely.geometry import box
        bbox = box(extent[0] - 1, extent[2] - 1, extent[1] + 1, extent[3] + 1)
        geoms = {}
        for name, category, feature in (("coast", "physical", "coastline"), ("border", "cultural", "admin_0_boundary_lines_land")):
            path = shpreader.natural_earth(resolution="10m", category=category, name=feature)
            geoms[name] = [g for g in shpreader.Reader(path).geometries() if g.intersects(bbox)]
        _geometry_cache[key] = geoms
    return _geometry_cache[key]


def draw_basemap(ax, extent: list[float], with_cities: bool = False) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    geoms = domain_geometries(extent)
    ax.add_geometries(geoms["coast"], ccrs.PlateCarree(), facecolor="none", edgecolor=INK2, linewidth=0.6, zorder=3)
    ax.add_geometries(geoms["border"], ccrs.PlateCarree(), facecolor="none", edgecolor=INK3, linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0, color=GRID,
                       xlocs=np.arange(-180, 181, 2), ylocs=np.arange(-90, 91, 2), zorder=1)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": INK2}
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS)
        spine.set_linewidth(0.8)
    if with_cities:
        halo = [pe.withStroke(linewidth=2.2, foreground=SURFACE)]
        for name, (x, y) in REFERENCE_CITIES.items():
            if not (extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]):
                continue
            right = x > extent[0] + 0.85 * (extent[1] - extent[0])  # keep the label on-axes near the right edge
            ax.plot(x, y, "s", ms=2.6, color=INK3, transform=ccrs.PlateCarree(), zorder=6)
            ax.text(x - 0.12 if right else x + 0.12, y + 0.06, name, fontsize=6.8, color=INK2,
                    ha="right" if right else "left", transform=ccrs.PlateCarree(), zorder=6, path_effects=halo)
