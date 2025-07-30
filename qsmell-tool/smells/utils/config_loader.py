import json
import os

# Compute absolute path to the project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.json')

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

def get_detector_option(smell_name, option, fallback=None):
    try:
        detector_config = CONFIG["Smells"][smell_name]["Detector"]
        return (
            detector_config.get("custom_values", {}).get(option)
            or detector_config.get("default_values", {}).get(option)
            or fallback
        )
    except KeyError:
        return fallback
