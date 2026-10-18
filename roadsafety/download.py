"""Find and download the accident files from data.go.th (a CKAN portal).

    python -m roadsafety.download --search "อุบัติเหตุ ทางหลวง"   # list candidate datasets
    python -m roadsafety.download --dataset <dataset-id>          # download its CSV/XLSX files

Files land in data/raw/. If the portal is unreachable you can download the files
in a browser from the dataset page and drop them into data/raw/ yourself.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

API = "https://data.go.th/api/3/action"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FORMATS = {"csv", "xls", "xlsx"}
HEADERS = {"User-Agent": "data-go-th-analysis (portfolio project)"}


def _call(action: str, **params) -> dict:
    resp = requests.get(f"{API}/{action}", params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(body.get("error"))
    return body["result"]


def search(query: str, rows: int = 20) -> list[dict]:
    result = _call("package_search", q=query, rows=rows)
    return [
        {
            "id": p["name"],
            "title": p.get("title"),
            "organization": (p.get("organization") or {}).get("title"),
            "formats": sorted({(r.get("format") or "").lower() for r in p.get("resources", [])} - {""}),
        }
        for p in result["results"]
    ]


def download(dataset_id: str, out_dir: Path = RAW_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    package = _call("package_show", id=dataset_id)
    saved = []
    for res in package["resources"]:
        fmt = (res.get("format") or Path(res["url"]).suffix.lstrip(".")).lower()
        if fmt not in FORMATS:
            continue
        name = re.sub(r"[^\w.\-]+", "_", res.get("name") or res["id"]).strip("_")
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--search", help="free-text search on data.go.th")
    g.add_argument("--dataset", help="dataset id (the last part of the dataset URL)")
    args = ap.parse_args()
    if args.search:
        for p in search(args.search):
            print(f"{p['id']:<40} {', '.join(p['formats']):<15} {p['organization']}  |  {p['title']}")
    else:
        download(args.dataset)


if __name__ == "__main__":
    main()
