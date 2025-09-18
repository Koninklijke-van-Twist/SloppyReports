import sys, os
from pathlib import Path

# add src/ to path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "src"))

from score import main

if __name__ == "__main__":
    main()