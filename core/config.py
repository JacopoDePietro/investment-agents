from pathlib import Path
import yaml

# Project root; all relative paths in config/settings.yaml are resolved against it

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    return cfg
