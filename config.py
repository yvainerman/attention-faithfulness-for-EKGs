"""
config.py
---------
Where the DATA lives (the code lives elsewhere).

Elsewhere (e.g. a laptop), point it at your own copy first:
    export ECG_DATA_ROOT=/path/to/folder_containing_ptbxl        (Linux / macOS)
    $env:ECG_DATA_ROOT = "C:/path/to/folder_containing_ptbxl"   (PowerShell)
"""
import os

DEFAULT_DATA_ROOT = "/home/yv33n/data"

DATA_ROOT = os.environ.get("ECG_DATA_ROOT") or DEFAULT_DATA_ROOT

DATA_PATHS = {
    "ptbxl": os.path.join(DATA_ROOT, "ptbxl"),
}


if __name__ == "__main__":
    # Quick check: python config.py
    print(f"ECG_DATA_ROOT = {os.environ.get('ECG_DATA_ROOT')!r}  ->  DATA_ROOT = {DATA_ROOT}")
    p = DATA_PATHS["ptbxl"]
    for name in ("ptbxl_database.csv", "scp_statements.csv", "records500"):
        ok = "OK     " if os.path.exists(os.path.join(p, name)) else "MISSING"
        print(f"  [{ok}] {os.path.join(p, name)}")