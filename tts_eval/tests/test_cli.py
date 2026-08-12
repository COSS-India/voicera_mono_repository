"""The CLI surface: it must faithfully wrap the API and return honest exit codes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _runs(tmp_path) -> str:
    return str(tmp_path / "runs")


def _list_json(main, runs, capsys) -> str:
    capsys.readouterr()  # drop prior output
    main(["list", "--runs", runs, "--json"])
    return capsys.readouterr().out


class TestCLI:
    def test_version_and_help_do_not_error(self, capsys):
        from tts_eval.cli import main

        for flags in (["--version"], ["--help"]):
            with pytest.raises(SystemExit) as exc:
                main(flags)
            assert exc.value.code == 0

    def test_no_command_is_a_usage_error(self):
        from tts_eval.cli import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_dataset_list(self, capsys):
        from tts_eval.cli import main

        assert main(["dataset", "list"]) == 0
        assert "indic_conversational_v1" in capsys.readouterr().out

    def test_run_list_report_roundtrip(self, tmp_path, capsys):
        from tts_eval.cli import main

        runs = _runs(tmp_path)
        assert main(["run", "-m", "mock", "-s", "smoke", "--runs", runs, "--quiet"]) == 0
        out = capsys.readouterr().out
        assert "success:     13/13" in out

        assert main(["list", "--runs", runs, "--json"]) == 0
        listing = json.loads(capsys.readouterr().out)
        assert listing and listing[0]["n_ok"] == 13
        run_id = listing[0]["run_id"]

        assert main(["report", run_id, "--runs", runs]) == 0
        assert (Path(runs) / run_id / "report.html").is_file()

    def test_verify_check_only_reproduces_fingerprint(self, tmp_path, capsys):
        from tts_eval.cli import main

        runs = _runs(tmp_path)
        main(["run", "-m", "mock", "-s", "smoke", "--runs", runs, "--quiet"])
        run_id = json.loads(_list_json(main, runs, capsys))[0]["run_id"]
        assert main(["verify", run_id, "--runs", runs, "--check-only"]) == 0
        assert "fingerprint reproduces" in capsys.readouterr().out

    def test_compare_incomparable_runs_exit_2(self, tmp_path, capsys):
        from tts_eval.cli import main

        runs = _runs(tmp_path)
        main(["run", "-m", "mock", "-s", "smoke", "--runs", runs, "--quiet"])
        main(["run", "-m", "mock", "-s", "latency", "--runs", runs, "--quiet"])
        ids = [r["run_id"] for r in json.loads(_list_json(main, runs, capsys))]
        # smoke vs latency: different dataset sample and concurrency -> blocked.
        assert main(["compare", ids[1], ids[0], "--runs", runs]) == 2

    def test_bad_model_is_a_clean_error_not_a_traceback(self, tmp_path, capsys):
        from tts_eval.cli import main

        code = main(["run", "-m", "no-such-model", "--runs", _runs(tmp_path), "--quiet"])
        assert code == 1
        assert "error:" in capsys.readouterr().err
