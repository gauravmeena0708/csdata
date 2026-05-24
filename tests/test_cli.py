from csdata.cli import main


def test_cli_list(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "adult" in out and "compas" in out


def test_cli_prepare_unknown_dataset_errors():
    rc = main(["prepare", "nope", "--out", "/tmp/x"])
    assert rc != 0
