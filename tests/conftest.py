"""Pytest configuration for the proxy test suite.

``pytest_addoption`` is only honored in a conftest.py at (or above) the
rootdir — pytest ignores it in a plain test module, which made the
``--live`` skipif condition raise "no option named '--live'" and error the
live roundtrip test on every run.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False,
        help="include the live HTTP roundtrip test (costs ~10 input tokens)"
    )
