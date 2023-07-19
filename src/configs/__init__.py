import importlib

__all__ = [
    'get_config'
]

def get_config(config_fn):
    config_module = importlib.import_module(f"configs.{config_fn}", package=None)
    return config_module.CFG