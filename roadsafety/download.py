"""Find and download the accident files from a Thai government CKAN portal.

    python -m roadsafety.download                                 # the dataset used in this project
    python -m roadsafety.download --search "อุบัติเหตุ"            # list candidate datasets
    python -m roadsafety.download --dataset <id> --portal https://data.go.th

The default is the Ministry of Transport's portal (datagov.mot.go.th), dataset
"roadaccident": one CSV per year, one row per accident. data.go.th runs the same
CKAN API, so --portal https://data.go.th works for its datasets too.

Files land in data/raw/. If the portal is unreachable you can download the files
in a browser from the dataset page and drop them into data/raw/ yourself.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

PORTAL = "https://datagov.mot.go.th"
DATASET = "roadaccident"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FORMATS = {"csv"}  # each year is also published as XLSX with the same rows
HEADERS = {"User-Agent": "data-go-th-analysis (portfolio project)"}


def _call(action: str, portal: str = PORTAL, **params) -> dict:
    resp = requests.get(f"{portal}/api/3/action/{action}", params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(body.get("error"))
    return body["result"]


def search(query: str, rows: int = 20, portal: str = PORTAL) -> list[dict]:
    result = _call("package_search", portal, q=query, rows=rows)
    return [
        {
            "id": p["name"],
            "title": p.get("title"),
            "organization": (p.get("organization") or {}).get("title"),
            "formats": sorted({(r.get("format") or "").lower() for r in p.get("resources", [])} - {""}),
        }
        for p in result["results"]
    ]


def download(dataset_id: str = DATASET, out_dir: Path = RAW_DIR, portal: str = PORTAL) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    package = _call("package_show", portal, id=dataset_id)
    saved = []
    for res in package["resources"]:
        fmt = (res.get("format") or Path(res["url"]).suffix.lstrip(".")).lower()
        if fmt not in FORMATS:
            continue
        name = Path(res["url"]).stem or re.sub(r"[^\w.\-]+", "_", res.get("name") or res["id"]).strip("_")
        path = out_dir / f"{name}.{fmt}"
        with requests.get(res["url"], headers=HEADERS, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"saved {path.relative_to(out_dir.parent.parent)}  ({path.stat().st_size / 1e6:.1f} MB)")
        saved.append(path)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--search", help="free-text search on the portal")
    g.add_argument("--dataset", default=DATASET, help="dataset id (the last part of the dataset URL)")
    ap.add_argument("--portal", default=PORTAL, help=f"CKAN portal base URL (default {PORTAL})")
    args = ap.parse_args()
    if args.search:
        for p in search(args.search, portal=args.portal):
            print(f"{p['id']:<40} {', '.join(p['formats']):<15} {p['organization']}  |  {p['title']}")
    else:
        download(args.dataset, portal=args.portal)


if __name__ == "__main__":
    main()
