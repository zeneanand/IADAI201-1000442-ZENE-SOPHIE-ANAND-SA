"""ParkVision AI — Root entrypoint for Streamlit Cloud deployment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
scripts_dir = ROOT / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from scripts.app import main

if __name__ == "__main__":
    main()
