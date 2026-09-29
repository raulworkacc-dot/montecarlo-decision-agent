import harness
import montecarlo_decisions


def test_packages_importable():
    assert harness.__version__ == "0.1.0"
    assert montecarlo_decisions.__version__ == "1.0.0"
