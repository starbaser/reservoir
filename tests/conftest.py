from reservoir import tsncd_bundle


def pytest_configure() -> None:
    tsncd_bundle.name_vendored_bundle_when_unset()
