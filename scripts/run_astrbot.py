"""Start AstrBot from this workspace with inherited resource limits."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0,str(root))
vendor = root/'data/searcher-vendor'
if vendor.is_dir(): sys.path.insert(0,str(vendor))
# Windows shells that predate setup do not inherit newly saved user variables.
# Load this one named key for the WorkBuddy plugin, without logging its value.
if os.name == 'nt' and not os.environ.get('WORKBUDDY_MASTER_KEY'):
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            value, _ = winreg.QueryValueEx(key, 'WORKBUDDY_MASTER_KEY')
        if isinstance(value, str) and value:
            os.environ['WORKBUDDY_MASTER_KEY'] = value
    except FileNotFoundError:
        pass
from assistant_security import install_log_redaction
install_log_redaction()
if (root/'data/plugins/astrbot_plugin_hapi_connector/main.py').is_file():
    from assistant_hapi_local import ensure_hub
    ensure_hub()  # failure leaves the assistant usable without HAPI
from astrbot.cli.__main__ import cli

if __name__ == "__main__":
    cli(["run"])
