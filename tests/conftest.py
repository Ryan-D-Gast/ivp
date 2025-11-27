"""Pytest configuration and fixtures for ivp tests."""
import pytest


@pytest.fixture
def num_parallel_threads():
    """Fixture to mimic SciPy's num_parallel_threads fixture.
    
    This fixture is used by tests that are copied from SciPy's test suite.
    In SciPy, this controls the number of threads for parallel operations.
    Since our implementation doesn't use parallel threads for these tests,
    we just return 1.
    """
    return 1
