"""Start AstrBot from this workspace with inherited resource limits."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0,str(root))
from assistant_security import install_log_redaction
install_log_redaction()
from astrbot.cli.__main__ import cli

if __name__ == "__main__":
    cli(["run"])
