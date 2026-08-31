from pathlib import Path


def test_dev_requirements_pin_pytest_without_changing_production_requirements():
    assert Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines() == ["pytest==9.1.1"]
    assert "pytest" not in Path("requirements.txt").read_text(encoding="utf-8").lower()
