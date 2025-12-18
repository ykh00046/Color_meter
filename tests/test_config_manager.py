import pytest

from src.data.config_manager import ConfigManager


def test_config_manager_init_data():
    cfg = ConfigManager(data={"a": 1})
    assert cfg.get("a") == 1


def test_config_manager_get_nested():
    cfg = ConfigManager(data={"a": {"b": 2}})
    assert cfg.get("a.b") == 2
    assert cfg.get("a.c", 3) == 3


def test_config_manager_set_nested():
    cfg = ConfigManager(data={})
    cfg.set("x.y", 10)
    assert cfg.get("x.y") == 10
