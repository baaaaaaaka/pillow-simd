import pytest

from PIL import __version__

pyroma = pytest.importorskip("pyroma", reason="Pyroma not installed")


def test_pyroma():
    # Arrange
    data = pyroma.projectdata.get_data(".")

    # Act
    rating = pyroma.ratings.rate(data)

    # Assert
    if "rc" in __version__:
        # Pyroma needs to chill about RC versions and not kill all our tests.
        assert rating == (
            9,
            ["The package's version number does not comply with PEP-386."],
        )

    else:
        # Allow the pyproject.toml advisory to reduce the score by 1.
        if rating[0] == 9 and any("pyproject.toml" in msg for msg in rating[1]):
            assert rating[0] == 9
        else:
            # Should have a perfect score
            assert rating == (10, [])
