from pathlib import Path

PROJECT_ROOT = Path("Data")
RAW_DATA = PROJECT_ROOT / "raw"
CLEAN_DATA = PROJECT_ROOT / "clean"
RESULT_DIR = PROJECT_ROOT / "results"

PROJECT_ROOT.mkdir(exist_ok=True)
RAW_DATA.mkdir(parents=True, exist_ok=True)
CLEAN_DATA.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
