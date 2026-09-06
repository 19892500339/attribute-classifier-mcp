"""Configuration manager for Attribute Classifier MCP Server."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


class Config:
    """Manages configuration with defaults and overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        if config_path:
            self._load_file(config_path)
    
    def _load_defaults(self):
        """Load default configuration."""
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
    
    def _load_file(self, path: str):
        """Load and merge a config file."""
        with open(path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        self._deep_merge(self._config, overrides)
    
    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path. E.g. 'training.epochs'"""
        keys = key_path.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set a config value by dot-separated path."""
        keys = key_path.split('.')
        target = self._config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dict."""
        return self._config.copy()
    
    def save(self, path: str):
        """Save current config to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


# Singleton config instance
_config: Optional[Config] = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create the singleton config instance."""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config
