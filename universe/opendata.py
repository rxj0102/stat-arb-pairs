"""Loader for the Kenneth R. French Data Library (free, no key).

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The study reads the frozen copy in data/snapshot. Run this file to download the data again:

    python -m universe.opendata

Downloads are cached as CSV under data/open_cache.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPENDATA_CACHE", ROOT / "data" / "open_cache"))
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
START = "1963-07-01"


def _get(url: str, tries: int = 5) -> bytes:
    import time
    last = None
    for k in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"})
            with urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # transient network errors
            last = e
            time.sleep(2 * (k + 1))
    raise last


def _cached(key: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{key}.csv"

def french(name: str, table: int = 0, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Ken French library table as decimal returns (percent / 100).

    name: file stem such as F-F_Research_Data_Factors_daily or 10_Industry_Portfolios_daily.
    table: which block in the CSV (0 = first, value weighted for portfolio files).
    """
    p = _cached(f"french_{name}_{table}")
    if not p.exists():
        raw = _get(FRENCH_URL.format(name=name))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        blocks, cur, header = [], [], None
        for line in text.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) > 1 and parts[0] == "" and header is None:
                header = parts
                continue
            if header is not None and parts and parts[0].isdigit() and len(parts[0]) in (6, 8):
                cur.append(parts)
            elif header is not None and cur:
                blocks.append((header, cur))
                header, cur = None, []
            elif header is not None and not parts[0]:
                continue
        if header is not None and cur:
            blocks.append((header, cur))
        hdr, rows = blocks[table]
        df = pd.DataFrame([r[1:len(hdr)] for r in rows], index=[r[0] for r in rows], columns=hdr[1:])
        df.to_csv(p)
    df = pd.read_csv(p, index_col=0, dtype={0: str})
    idx = df.index.astype(str)
    fmt = "%Y%m%d" if len(idx[0]) == 8 else "%Y%m"
    df.index = pd.to_datetime(idx, format=fmt)
    if fmt == "%Y%m":
        df.index = df.index + pd.offsets.MonthEnd(0)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.mask(df <= -99.99) / 100.0
    return df.loc[start:end]


def refresh_snapshot() -> Path:
    """Download the 49 industry daily returns (value weighted) and rewrite the snapshot."""
    df = french("49_Industry_Portfolios_daily", table=0, start=START)
    df.index.name = "date"
    out = ROOT / "data" / "snapshot" / "industry_returns.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, compression={"method": "gzip", "mtime": 0})
    return out


if __name__ == "__main__":
    print(refresh_snapshot())
