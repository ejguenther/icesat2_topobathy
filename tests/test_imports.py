"""Smoke test: every phoreal module imports cleanly."""
import importlib
import pkgutil

import pytest

import phoreal

MODULES = [m.name for m in pkgutil.walk_packages(phoreal.__path__, prefix="phoreal.")]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
