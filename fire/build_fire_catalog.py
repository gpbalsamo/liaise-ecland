#!/usr/bin/env python3
"""Build the committed fire-event catalogue used to validate LEFIRE (fire/data/catalog.json).

Combines two sources into one flat, small, LIAISE-domain-filtered list of notable fires:

  1. EFFIS (2000-2024): every record from fire/fetch_effis_fires.py's raw pull whose
     centroid falls inside the domain (read from --surfclim's own grid, so this stays
     correct if the domain ever changes) and whose burned area clears --min-area-ha.
  2. A small, hand-researched set of larger pre-2000 fires (EFFIS has no data before
     2000), each with its own press/agency source noted -- this list is NOT exhaustive;
     it is only the handful of large fires actually looked up, not a complete record of
     1989-1999 fire activity in the domain. Extend it here as more are researched.

Domain generality: the bounding box is derived from --surfclim's own lat/lon grid, not
hard-coded, so this script works unmodified if the domain (surfclim/soilinit) changes
for a different regional application of this repo.

Output schema (one dict per fire): key, name, date, lon, lat, area_ha, size, extra,
src, origin ("EFFIS" or "curated"). `size` is deliberately a free-text field: reported
burned area often varies across sources by tens of percent, and pinning to a single
EFFIS number would silently discard that spread for the curated fires.

Usage
-----
    python3 build_fire_catalog.py --effis-json fire/work/effis_raw.json \\
        --surfclim init_clim/work/surfclim --min-area-ha 1000 --out fire/data/catalog.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# Hand-researched fires before EFFIS's 2000 start, >= 1000 ha, found while validating LEFIRE
# on 2026-09-21/22 -- see each entry's `src`. Positions are municipality-level (+-0.2 deg);
# `area_ha` is a representative point value for sizing map markers, `size` carries the full
# reported range/context. Add more here as they are researched; there is no programmatic
# source for this period.
CURATED_PRE_2000 = [
    dict(name="Cerbère–Banyuls fire, Pyrénées-Orientales (FR)", date="2023-04-16",
         lon=3.17, lat=42.44, area_ha=1000,
         size="930–1,000+ ha (press reports); EFFIS itself maps only 770 ha here",
         extra="tramontane gusts to 120 km/h; multi-year coastal drought",
         src="Préfecture des Pyrénées-Orientales; Copernicus"),
    dict(name="Sainte-Victoire, Saint-Marc-Jaumegarde, Bouches-du-Rhône (FR)", date="1989-08-28",
         lon=5.49, lat=43.55, area_ha=5000, size="~5,000 ha (reported)",
         extra="mistral and exceptional drought; on the eastern edge of the domain",
         src="Les Amis de Sainte-Victoire; Maritima"),
    dict(name="Salaunes–Carcans, Gironde (FR)", date="1990-03-31",
         lon=-0.98, lat=44.98, area_ha=5200, size="5,200 ha (reported), in one day",
         extra="very dry winter after the 1989 drought",
         src="Canopée; Assemblée nationale"),
    dict(name="Villarluengo–Espadilla (Maestrazgo), Teruel/Castellón (ES)", date="1994-07-02",
         lon=-0.42, lat=40.57, area_ha=19310, size="18,000–19,310 ha (reported)",
         extra='start of the July 1994 "black week" in Valencia/Aragon',
         src="El Periodic; El Español Aragón"),
    dict(name="Catalonia fires (Bages, Berguedà, Solsonès) (ES)", date="1994-07-04",
         lon=1.90, lat=41.90, area_ha=62500, size="45,000–80,000 ha (reported, 4–8 July)",
         extra="5 deaths; central Catalonia",
         src="Wikidata; Catalan News; CTFC"),
    dict(name="Millares, Valencia (ES)", date="1994-07-04",
         lon=-0.97, lat=39.15, area_ha=25430, size="25,430 ha (reported)",
         extra="July 1994; 138,400 ha burned in the region that year",
         src="El Periodic; Noticias CV"),
    dict(name="Requena, Valencia (ES)", date="1994-07-05",
         lon=-1.10, lat=39.49, area_ha=24064, size="24,064 ha (reported)",
         extra='July 1994 "black week"',
         src="El Periodic; Noticias CV"),
    dict(name="Central Catalonia (Solsonès, Bages, Segarra) (ES)", date="1998-07-18",
         lon=1.60, lat=41.95, area_ha=27000, size="~27,000 ha (reported)",
         extra="largest Catalan fire since 1994",
         src="CTFC blog; Wikipedia"),
]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--effis-json", type=Path, required=True, help="output of fetch_effis_fires.py")
    p.add_argument("--surfclim", type=Path, default=Path("init_clim/work/surfclim"),
                    help="defines the domain box (lat/lon grid extent, +- half a cell); "
                         "default matches the committed LIAISE surfclim")
    p.add_argument("--min-area-ha", type=float, default=1000.0, help="EFFIS fires below this are dropped")
    p.add_argument("--end-date", default="2024-12-31", help="drop EFFIS fires after this date "
                    "(keep in sync with the forcing/run archive's own end year)")
    p.add_argument("--out", type=Path, default=Path("fire/data/catalog.json"))
    return p.parse_args()


def domain_box(surfclim: Path) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) cell edges of surfclim's own regular grid."""
    with Dataset(surfclim) as d:
        lat = np.ma.filled(d["lat"][:], np.nan).astype("f8")
        lon = np.ma.filled(d["lon"][:], np.nan).astype("f8")
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    return (lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2)


def tidy_commune(name: str) -> str:
    """'Atalaya, La' -> 'La Atalaya' (EFFIS puts the article after the comma)."""
    m = re.match(r"^(.*), (La|El|Las|Los|Els|Les|Lo|Le|L')$", name.strip())
    if not m:
        return name.strip()
    article, base = m.group(2), m.group(1)
    return f"{article}{'' if article.endswith(chr(39)) else ' '}{base}"


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def main() -> None:
    args = get_args()
    lon0, lon1, lat0, lat1 = domain_box(args.surfclim)
    raw = json.load(args.effis_json.open())

    events = []
    for r in raw:
        date = r["firedate"][:10]
        if not (lon0 <= r["lon"] <= lon1 and lat0 <= r["lat"] <= lat1):
            continue
        if date > args.end_date or r["area_ha"] < args.min_area_ha:
            continue
        name = f"{tidy_commune(r['commune'])}, {r['province']} ({r['country']})"
        events.append(dict(
            key=f"{date}_{slugify(r['commune'])}_{r['id']}", name=name, date=date,
            lon=round(r["lon"], 3), lat=round(r["lat"], 3), area_ha=r["area_ha"],
            size=f"{r['area_ha']:,} ha (EFFIS)", extra="EFFIS mapped burnt area; date = first satellite detection",
            src="EFFIS (Copernicus)", origin="EFFIS",
        ))

    for m in CURATED_PRE_2000:
        events.append(dict(m, key=f"{m['date']}_{slugify(m['name'])[:30]}", origin="curated"))

    events.sort(key=lambda e: e["date"])
    keys = [e["key"] for e in events]
    assert len(keys) == len(set(keys)), "duplicate event key -- two fires collided, check the EFFIS id/curated slug"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(events, args.out.open("w"), ensure_ascii=False, indent=0)

    n_effis = sum(e["origin"] == "EFFIS" for e in events)
    n_curated = sum(e["origin"] == "curated" for e in events)
    print(f"domain box: lon [{lon0:.2f}, {lon1:.2f}], lat [{lat0:.2f}, {lat1:.2f}]")
    print(f"events: {len(events)} (EFFIS {n_effis}, curated {n_curated}) -> {args.out}")


if __name__ == "__main__":
    main()
