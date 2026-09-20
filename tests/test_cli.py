import pytest

from covertype.cli import build_parser, main


def test_defaults_use_the_synthetic_dataset():
    args = build_parser().parse_args([])
    assert not args.csv and not args.uci


def test_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--csv", "a.csv", "--uci"])


def test_a_run_prints_the_imbalance_and_the_table(capsys):
    assert (
        main(["--rows", "5000", "--train-sizes", "500,1500", "--test-size", "1000"]) == 0
    )
    out = capsys.readouterr().out
    assert "imbalance" in out
    assert "majority-class accuracy" in out
    assert "macro-F1" in out


def test_an_infeasible_svm_run_is_refused_before_it_starts(capsys):
    assert main(["--models", "svm-rbf", "--train-sizes", "200000"]) == 2
    assert "refusing svm-rbf" in capsys.readouterr().err


def test_an_unknown_model_is_refused(capsys):
    assert main(["--models", "naive-bayes"]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_non_integer_train_sizes_are_refused(capsys):
    assert main(["--train-sizes", "a,b"]) == 2
    assert "must be integers" in capsys.readouterr().err


def test_a_missing_csv_is_a_usage_error(tmp_path, capsys):
    assert main(["--csv", str(tmp_path / "absent.csv")]) == 2
    assert "file not found" in capsys.readouterr().err


def test_a_malformed_csv_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("A,B\n1,2\n")
    assert main(["--csv", str(path)]) == 2
    assert "Cover_Type" in capsys.readouterr().err
