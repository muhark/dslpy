from __future__ import annotations

import importlib.metadata

import dsl as m


def test_version():
    assert importlib.metadata.version("dsl") == m.__version__
