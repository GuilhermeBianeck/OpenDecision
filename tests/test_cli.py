"""CLI smoke checks and clean errors without model downloads."""

import json

import pytest

from opendecision.cli import build_parser, main


def test_models_and_doctor(capsys):
    assert main(["models"]) == 0
    assert any(model["name"] == "base" for model in json.loads(capsys.readouterr().out))
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["python"]
    assert isinstance(report["mps_available"], bool)
    if report["mps_available"] and not report["cuda_available"]:
        assert report["recommended_device"] == "mps"


@pytest.mark.parametrize("command", ["decide", "rank", "boolean"])
def test_local_demo_decision(command, capsys):
    args = [command, "--model", "demo", "--state", "billing support", "--question", "Which team?"]
    if command != "boolean":
        args.extend(["--choices", "billing", "technical"])
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert ("decision" in result) if command != "decide" else (result["choice"] == "billing")


def test_score_command(capsys):
    args = [
        "score",
        "--model",
        "demo",
        "--state",
        "billing outage for every customer",
        "--question",
        "How severe?",
        "--levels",
        "cosmetic",
        "degraded",
        "outage for customers",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["legend"]["2"] == "outage for customers"
    assert result["level"] == 2
    assert main([*args[:-3], "only-one"]) == 1
    assert "levels" in capsys.readouterr().err


def test_boolean_unsupported_threshold_requires_statement_backend(capsys):
    args = ["boolean", "--model", "demo", "--state", "x", "--question", "claim"]
    assert main([*args, "--unsupported-threshold", "0.5"]) == 1
    assert "statement scoring" in capsys.readouterr().err
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "binary_choice"


def test_private_state_file(tmp_path, capsys):
    path = tmp_path / "context.txt"
    path.write_text("billing issue", encoding="utf-8")
    assert (
        main(
            [
                "decide",
                "--model",
                "demo",
                "--state-file",
                str(path),
                "--question",
                "Team?",
                "--choices",
                "billing",
                "technical",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["choice"] == "billing"


def test_json_state_and_option_files(tmp_path, capsys):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"message": "billing question", "priority": 2}))
    choices = tmp_path / "choices.json"
    choices.write_text(
        json.dumps([{"label": "billing", "description": "invoices and payments"}, "technical"])
    )
    args = ["decide", "--model", "demo", "--state-json", str(state), "--question", "Team?"]
    assert main([*args, "--choices-json", str(choices)]) == 0
    assert json.loads(capsys.readouterr().out)["choice"] == "billing"
    state.write_text(json.dumps("just text"))
    assert main([*args, "--choices", "billing", "technical"]) == 1
    assert "--state-json" in capsys.readouterr().err
    state.write_text(json.dumps(["billing question", "priority two"]))
    choices.write_text(json.dumps({"label": "not a list"}))
    assert main([*args, "--choices-json", str(choices)]) == 1
    assert "--choices-json" in capsys.readouterr().err


def test_ask_command(tmp_path, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "team": {"type": "choice", "question": "Team?", "choices": ["billing", "other"]},
                "urgent": {"type": "boolean", "statement": "billing"},
            }
        )
    )
    args = ["ask", "--model", "demo", "--state", "billing issue", "--questions-json"]
    assert main([*args, str(questions)]) == 0
    answers = json.loads(capsys.readouterr().out)
    assert answers["team"]["choice"] == "billing"
    assert answers["urgent"]["type"] == "boolean"
    questions.write_text(json.dumps([]))
    assert main([*args, str(questions)]) == 1
    assert "--questions-json" in capsys.readouterr().err


def test_bad_model_clean_error(capsys):
    assert (
        main(
            [
                "decide",
                "--model",
                "missing",
                "--state",
                "",
                "--question",
                "?",
                "--choices",
                "a",
                "b",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert not captured.out
    assert "Unknown model" in captured.err
    assert "Traceback" not in captured.err


def test_cli_invalid_choices(capsys):
    assert (
        main(["decide", "--model", "demo", "--state", "", "--question", "?", "--choices", "a", "a"])
        == 1
    )
    assert "unique" in capsys.readouterr().err


def test_local_server_default_and_explicit_network_warning(monkeypatch, capsys):
    pytest.importorskip("uvicorn")
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    assert main(["serve", "--model", "demo"]) == 0
    assert calls[-1]["host"] == "127.0.0.1"
    assert calls[-1]["port"] == 8042
    assert not capsys.readouterr().err
    assert main(["serve", "--model", "demo", "--host", "0.0.0.0"]) == 0
    assert "unauthenticated" in capsys.readouterr().err


def test_invalid_server_port(capsys):
    pytest.importorskip("uvicorn")
    assert main(["serve", "--port", "0"]) == 1
    assert "--port" in capsys.readouterr().err


def test_calibration_split_is_fixed():
    parser = build_parser()
    args = parser.parse_args(["calibrate", "--dataset", "some.jsonl", "--output", "profile.json"])
    assert not hasattr(args, "split")


def test_calibration_uses_only_labeled_calibration_split(tmp_path, capsys):
    from opendecision.calibration import CalibrationProfile

    rows = []
    for index, (split, family) in enumerate(
        [
            ("train", "objective"),
            ("calibration", "objective"),
            ("calibration", "agent_control"),
            ("calibration", "ambiguous"),
            ("calibration", "subjective"),
            ("validation", "objective"),
            ("test", "objective"),
        ]
    ):
        rows.append(
            {
                "id": f"case-{index}",
                "group_id": f"group-{index}",
                "split": split,
                "family": family,
                "state": "billing",
                "question": "Team?",
                "choices": ["billing", "technical"],
                "target": "billing",
                "reference_policy": "billing first" if family == "subjective" else None,
            }
        )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "profiles" / "demo.json"
    assert (
        main(["calibrate", "--model", "demo", "--dataset", str(dataset), "--output", str(output)])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["examples"] == 2
    assert report["candidate_counts"] == {"2": 2}
    # Two rows are below min_rows_per_count, so only the pooled temperature is fitted.
    assert report["profile"]["temperatures"] is None
    assert report["validation"]["before"]["count"] == 1
    assert report["validation"]["after"]["accuracy"] == report["validation"]["before"]["accuracy"]
    profile = CalibrationProfile.load(output)
    assert profile.split == "calibration"
    assert profile.sample_count == 2
    assert profile.dataset_sha256
    assert profile.precision == "float64"


def test_calibration_without_validation_rows_reports_no_held_out_effect(tmp_path, capsys):
    rows = [
        {
            "id": f"case-{index}",
            "group_id": f"group-{index}",
            "split": "calibration",
            "family": "objective",
            "state": "billing",
            "question": "Team?",
            "choices": ["billing", "technical"],
            "target": "billing",
        }
        for index in range(3)
    ]
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "profile.json"
    assert (
        main(
            [
                "calibrate",
                "--model",
                "demo",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--no-validation",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["validation"] is None


def test_calibration_refuses_test_only_data(tmp_path, capsys):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "test",
                "group_id": "test",
                "split": "test",
                "family": "objective",
                "state": "billing",
                "question": "Team?",
                "choices": ["billing", "technical"],
                "target": "billing",
            }
        )
    )
    output = tmp_path / "never-created.json"
    assert (
        main(["calibrate", "--model", "demo", "--dataset", str(dataset), "--output", str(output)])
        == 1
    )
    assert "calibration split" in capsys.readouterr().err
    assert not output.exists()


def test_demo_benchmark_writes_report(tmp_path, capsys):
    output = tmp_path / "demo"
    assert (
        main(
            [
                "benchmark",
                "--model",
                "demo",
                "--dataset",
                "benchmarks/datasets/core.jsonl",
                "--output",
                str(output),
                "--limit",
                "3",
            ]
        )
        == 0
    )
    reports = json.loads(capsys.readouterr().out)
    assert reports["reports"][0]["model"] == "local:demo"
    assert output.with_suffix(".json").is_file()
    assert output.with_suffix(".md").is_file()
