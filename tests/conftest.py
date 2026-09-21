"""Shared pytest configuration for the CPU tier.

Every test runs with a private copy of ``os.environ``: the planner and
``SessionProfile.apply()`` read (and some tests deliberately set) ``PLAN_*`` /
``CHAIN`` / ``FHE_*`` knobs, and without this snapshot a test's exports would leak
into the next module in collection order.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_environ():
    """Snapshot ``os.environ`` before each test and restore it afterwards (also for
    ``unittest.TestCase`` methods, which pytest runs through the same fixture hooks)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
