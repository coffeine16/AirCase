def test_native_package_imports():
    import intelligence.models.forecast.native as native
    assert native is not None


def test_numba_is_importable():
    import numba
    assert numba.__version__
