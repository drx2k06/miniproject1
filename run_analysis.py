"""
Top-level entrypoint for MicroQuant market microstructure analysis pipeline.

Author: Daniel Richard Xavier
Usage:
    python run_analysis.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.cli import run_pipeline

if __name__ == "__main__":
    run_pipeline()
