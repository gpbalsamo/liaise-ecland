#!/usr/bin/env python3
"""Pull burnt-area events from the EU's EFFIS database (Copernicus), attributes only.

EFFIS (European Forest Fire Information System) publishes a public REST API with no
authentication needed, covering fires mapped from 2000 onward. This script keeps only
the small attribute fields (location, date, burned area) and drops the burnt-area
polygons, which this project has no use for and would make the output far larger.

This is a *raw* pull across whichever countries are requested -- not filtered to any
particular region -- so its output should not be committed (see fire/build_fire_catalog.py,
which does the LIAISE-domain filtering and produces the small, committed reference file).

Usage
-----
    python3 fetch_effis_fires.py --countries ES FR AD --min-area-ha 500 \\
        --out fire/work/effis_raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://api.effis.emergency.copernicus.eu/rest/2/burntareas/current/"
KEEP_FIELDS = ("id", "country", "province", "commune", "firedate", "lastfiredate", "area_ha")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", nargs="+", default=["ES", "FR", "AD"],
                    help="ISO 3166-1 alpha-2 country codes to query (default: Spain, France, Andorra -- "
                         "covers the LIAISE domain's box with margin)")
    p.add_argument("--min-area-ha", type=float, default=500.0,
                    help="EFFIS-side filter (area_ha__gte); keep this at or below whatever "
                         "fire/build_fire_catalog.py's --min-area-ha will use downstream")
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("fire/work/effis_raw.json"),
                    help="raw attribute dump, one record per fire (default: fire/work/, not committed)")
    return p.parse_args()


def fetch_country(country: str, min_area_ha: float, page_size: int, retries: int) -> list[dict]:
    out: list[dict] = []
    url = f"{BASE_URL}?country={country}&area_ha__gte={min_area_ha}&limit={page_size}&ordering=firedate"
    while url:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url.replace("http://", "https://"), timeout=60) as r:
                    d = json.load(r)
                break
            except Exception as e:  # noqa: BLE001 -- transient network/HTTP errors, retried
                print(f"retry {country} attempt {attempt}: {e}", file=sys.stderr, flush=True)
                time.sleep(3)
        else:
            sys.exit(f"failed to fetch {url} after {retries} attempts")
        for r in d["results"]:
            lon, lat = r["centroid"]["coordinates"]
            out.append({k: r[k] for k in KEEP_FIELDS} | {"lon": lon, "lat": lat})
        print(f"{country}: {len(out)} of {d['count']}", flush=True)
        url = d["next"]
    return out


def main() -> None:
    args = get_args()
    out: list[dict] = []
    for cc in args.countries:
        out.extend(fetch_country(cc, args.min_area_ha, args.page_size, args.retries))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, args.out.open("w"))
    print(f"saved {len(out)} records -> {args.out}")


if __name__ == "__main__":
    main()
