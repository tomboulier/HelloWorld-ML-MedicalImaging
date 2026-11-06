import pytest

from hello_world_medical_imaging.cli import parse_args


def test_version_flag_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
