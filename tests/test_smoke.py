import harness


def test_package_importable():
    assert harness.__version__ == "0.1.0"
