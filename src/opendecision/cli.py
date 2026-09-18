"""Command-line interface for explicit downloads and otherwise offline inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default))


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default="base", help="Model tier or registered model (default: base)"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--calibration", type=Path, help="Matching calibration profile JSON")
    parser.add_argument("--batch-size", type=int, default=32, help="Candidate microbatch size")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum tokenized input length (default: model context, capped at 2048)",
    )
    parser.add_argument("--template", default="default", choices=("default", "short"))
    parser.add_argument(
        "--precision",
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="Default: bfloat16 on a GPU, float32 on CPU",
    )


def _request_options(
    parser: argparse.ArgumentParser, *, candidates: str | None = "choices"
) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--state", help="Context text (use --state-file for large or private input)"
    )
    source.add_argument(
        "--state-file", type=Path, help="UTF-8 context file; '-' reads standard input"
    )
    source.add_argument(
        "--state-json",
        type=Path,
        help="JSON file holding a record (object) or list of texts to use as the state",
    )
    parser.add_argument("--question", required=True)
    choices = candidates is not None
    if candidates == "choices":
        options = parser.add_mutually_exclusive_group(required=True)
        options.add_argument("--choices", nargs="+", help="Candidate labels")
        options.add_argument(
            "--choices-json",
            type=Path,
            help="JSON file: a list of labels or of {label, description, not_for, examples}",
        )
    elif candidates == "levels":
        parser.add_argument(
            "--levels", nargs="+", required=True, help="Rubric levels from lowest to highest"
        )
    if choices:
        parser.add_argument("--raw-scores", action="store_true")
    parser.add_argument("--abstain-threshold", type=float, help="Minimum top probability")
    parser.add_argument("--margin-threshold", type=float, help="Minimum top-two probability margin")
    if not choices:
        parser.add_argument(
            "--unsupported-threshold",
            type=float,
            help="Abstain when p(state settles neither way) exceeds this; NLI backends only",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opendecision",
        description="Local decision scoring. Downloads only happen with the explicit pull command.",
    )
    parser.add_argument("--version", action="version", version="OpenDecision 0.1.0a1")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List pinned model identities and licenses")
    pull = commands.add_parser("pull", help="Download and smoke-test a model (requires network)")
    pull.add_argument("name", nargs="?", default="base")
    pull.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda", "auto"))
    commands.add_parser("doctor", help="Inspect local hardware and optional runtimes")
    for name in ("decide", "rank", "boolean", "score"):
        command = commands.add_parser(name, help=f"Run local {name} inference")
        _model_options(command)
        _request_options(
            command,
            candidates=None if name == "boolean" else "levels" if name == "score" else "choices",
        )
    ask = commands.add_parser("ask", help="Answer several typed questions about one state")
    _model_options(ask)
    source = ask.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="Context text")
    source.add_argument("--state-file", type=Path, help="UTF-8 context file; '-' reads stdin")
    source.add_argument("--state-json", type=Path, help="JSON record or list of texts")
    ask.add_argument(
        "--questions-json",
        type=Path,
        required=True,
        help='JSON object of id -> {"type": "choice"|"boolean"|"score", ...}',
    )
    serve = commands.add_parser("serve", help="Start the local HTTP server")
    _model_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8042, type=int)
    serve.add_argument("--max-concurrency", default=4, type=int)
    benchmark = commands.add_parser(
        "benchmark", help="Evaluate a labeled dataset and save a report"
    )
    _model_options(benchmark)
    benchmark.add_argument("--models", help="Comma-separated local:base,local:smart,gemini[:model]")
    benchmark.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/core.jsonl"))
    benchmark.add_argument("--output", type=Path, default=Path("benchmarks/reports/latest"))
    benchmark.add_argument("--split", default="test", choices=("calibration", "validation", "test"))
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--robustness", action="store_true")
    benchmark.add_argument("--request-batch-size", type=int, default=1)
    calibrate = commands.add_parser(
        "calibrate", help="Fit temperature using only the calibration split"
    )
    _model_options(calibrate)
    calibrate.add_argument("--dataset", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument(
        "--pooled",
        action="store_true",
        help="Fit one temperature for every candidate count instead of one per count",
    )
    calibrate.add_argument(
        "--min-rows-per-count",
        type=int,
        default=25,
        help="Candidate counts with fewer calibration rows fall back to the pooled temperature",
    )
    calibrate.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip scoring the validation split; the report then shows no held-out effect",
    )
    return parser


def _load_model(args: argparse.Namespace, *, name: str | None = None):
    from opendecision import DecisionModel
    from opendecision.calibration import CalibrationProfile

    profile = CalibrationProfile.load(args.calibration) if args.calibration else None
    return DecisionModel(
        name or args.model,
        device=args.device,
        calibration=profile,
        batch_size=args.batch_size,
        max_length=args.max_length,
        template=args.template,
        precision=args.precision,
    )


def _state(args: argparse.Namespace) -> Any:
    if args.state is not None:
        return args.state
    if getattr(args, "state_json", None) is not None:
        loaded = json.loads(args.state_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, (dict, list)):
            raise ValueError("--state-json must contain a JSON object or a list of texts")
        return loaded
    if str(args.state_file) == "-":
        return sys.stdin.read()
    return args.state_file.read_text(encoding="utf-8")


def _choices(args: argparse.Namespace) -> list[Any]:
    if args.choices is not None:
        return args.choices
    loaded = json.loads(args.choices_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("--choices-json must contain a JSON list")
    return loaded


def doctor() -> dict[str, Any]:
    """Inspect local runtime availability without downloading any model."""
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "ram_gib": None,
        "torch": None,
        "mps_available": False,
        "cuda_available": False,
        "onnxruntime_available": importlib.util.find_spec("onnxruntime") is not None,
        "recommended_device": "cpu",
        "recommended_precision": "float32",
        "recommended_backend": "base",
        "recommendation": "Start with base; device='auto' selects CUDA, then MPS, then CPU.",
    }
    try:
        import psutil

        report["ram_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        try:
            report["ram_gib"] = round(
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 2
            )
        except (ValueError, OSError, AttributeError):
            if platform.system() == "Darwin":
                try:
                    report["ram_gib"] = round(
                        int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
                        / 1024**3,
                        2,
                    )
                except (OSError, ValueError, subprocess.CalledProcessError):
                    pass
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            report["torch"] = torch.__version__
            report["mps_available"] = bool(torch.backends.mps.is_available())
            report["cuda_available"] = bool(torch.cuda.is_available())
            if report["cuda_available"]:
                report["recommended_device"] = "cuda"
            elif report["mps_available"]:
                report["recommended_device"] = "mps"
            from opendecision.backends.transformers import default_precision

            report["recommended_precision"] = default_precision(report["recommended_device"])
        except (ImportError, OSError, RuntimeError) as error:
            report["torch_error"] = str(error)
    else:
        report["recommendation"] = "Install opendecision[inference] for real model inference."
    return report


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from opendecision.benchmark import load_dataset, run_benchmark, write_report

    cases = load_dataset(args.dataset)
    names = [part.strip() for part in (args.models or f"local:{args.model}").split(",")]
    if not names or any(not name for name in names):
        raise ValueError("--models must contain nonempty model names")
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for name in names:
        if name.startswith("local:"):
            engine = _load_model(args, name=name.removeprefix("local:"))
        elif name == "gemini" or name.startswith("gemini:"):
            from opendecision.external import ExternalUnavailable, create_external_model

            try:
                engine = create_external_model(name)
            except ExternalUnavailable as error:
                skipped.append({"model": name, "reason": str(error)})
                continue
        else:
            raise ValueError(f"Unknown provider {name!r}; use local:<model> or gemini[:model]")
        report = run_benchmark(
            engine,
            cases,
            batch_size=args.request_batch_size,
            split=args.split,
            limit=args.limit,
            robustness=args.robustness,
        )
        output = args.output
        if len(names) > 1:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
            output = output.with_name(f"{output.stem}-{safe_name}")
        reports.append({"model": name, "files": write_report(report, output)})
    return {"reports": reports, "skipped": skipped}


def _calibration_rows(engine: Any, cases: list[Any]) -> tuple[list[list[float]], list[int]]:
    """Score labeled choice cases in batches and return raw score rows with label indexes."""
    from opendecision.schemas import DecisionRequest

    scores, targets = [], []
    for offset in range(0, len(cases), 32):
        chunk = cases[offset : offset + 32]
        results = engine.choose_batch(
            [
                DecisionRequest(
                    state=case.state,
                    question=case.question,
                    choices=case.choices,
                    include_raw_scores=True,
                )
                for case in chunk
            ]
        )
        for case, result in zip(chunk, results):
            if result.raw_scores is None:
                raise ValueError("Backend did not expose scores required for calibration")
            scores.append([result.raw_scores[choice] for choice in case.choices])
            targets.append(case.choices.index(case.target))
    return scores, targets


def _held_out_effect(profile: Any, scores: list[list[float]], targets: list[int]) -> dict[str, Any]:
    """Calibration metrics on rows the profile was not fitted on."""
    from opendecision.confidence import softmax
    from opendecision.metrics import classification_metrics

    def measure(transform: Any) -> dict[str, Any]:
        records = [
            {
                "target": str(target),
                "probabilities": {str(i): p for i, p in enumerate(transform(row))},
                "choice": str(max(range(len(row)), key=lambda i: transform(row)[i])),
                "confidence": 0.0,
                "abstained": False,
            }
            for row, target in zip(scores, targets)
        ]
        metrics = classification_metrics(records)
        return {
            key: metrics[key]
            for key in ("count", "accuracy", "ece", "negative_log_likelihood", "brier_score")
        }

    return {
        "before": measure(softmax),
        "after": measure(profile.transform),
        "note": "Temperature scaling never changes candidate ordering, so accuracy is identical.",
    }


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    from opendecision.benchmark import load_dataset
    from opendecision.calibration import fit_temperature

    if args.calibration is not None:
        raise ValueError(
            "Fit a new profile without --calibration; existing calibration cannot be chained"
        )

    # Calibration fits the candidate softmax; boolean statements and rubric levels
    # are scored through different paths and are excluded until profiled separately.
    def labeled(split: str) -> list[Any]:
        return [
            case
            for case in dataset
            if case.split == split
            and case.kind == "choice"
            and case.family in {"objective", "agent_control", "robustness"}
            and case.target is not None
        ]

    dataset = load_dataset(args.dataset)
    cases = labeled("calibration")
    if not cases:
        raise ValueError("Dataset has no labeled objective examples in the calibration split")
    engine = _load_model(args)
    scores, targets = _calibration_rows(engine, cases)
    backend = engine.backend
    profile = fit_temperature(
        scores,
        targets,
        backend=backend.name,
        model=backend.model_id,
        revision=backend.revision,
        template=args.template,
        max_length=engine.max_length,
        precision=backend.precision,  # resolved during the scoring above
        task_family="objective_agent_control",
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        per_choice_count=not args.pooled,
        min_rows_per_count=args.min_rows_per_count,
    )
    profile.save(args.output)
    report: dict[str, Any] = {
        "output": str(args.output),
        "examples": len(cases),
        "split": "calibration",
        "candidate_counts": dict(sorted(Counter(len(row) for row in scores).items())),
        "profile": profile.model_dump(mode="json"),
    }
    validation = [] if args.no_validation else labeled("validation")
    if validation:
        held_out_scores, held_out_targets = _calibration_rows(engine, validation)
        report["validation"] = _held_out_effect(profile, held_out_scores, held_out_targets)
    else:
        report["validation"] = None
    return report


def run(args: argparse.Namespace) -> Any:
    if args.command == "models":
        from opendecision.registry import list_models

        return list_models()
    if args.command == "pull":
        from opendecision.registry import pull_model

        return pull_model(args.name, device=args.device)
    if args.command == "doctor":
        return doctor()
    if args.command in {"decide", "rank", "boolean", "score"}:
        engine = _load_model(args)
        request: dict[str, Any] = {
            "state": _state(args),
            "question": args.question,
            "abstain_threshold": args.abstain_threshold,
            "margin_threshold": args.margin_threshold,
        }
        if args.command == "boolean":
            request["unsupported_threshold"] = args.unsupported_threshold
        elif args.command == "score":
            request.update(levels=args.levels, include_raw_scores=args.raw_scores)
        else:
            request.update(choices=_choices(args), include_raw_scores=args.raw_scores)
        return getattr(engine, "choose" if args.command == "decide" else args.command)(**request)
    if args.command == "ask":
        questions = json.loads(args.questions_json.read_text(encoding="utf-8"))
        if not isinstance(questions, dict):
            raise ValueError("--questions-json must contain a JSON object keyed by question id")
        return _load_model(args).ask(state=_state(args), questions=questions)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "serve":
        import uvicorn

        from opendecision.calibration import CalibrationProfile
        from opendecision.server import create_app

        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            print(
                "WARNING: binding outside localhost exposes an unauthenticated inference API. "
                "Use a trusted network or add authentication and TLS at a reverse proxy.",
                file=sys.stderr,
            )
        profile = CalibrationProfile.load(args.calibration) if args.calibration else None
        app = create_app(
            args.model,
            args.device,
            profile,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            max_length=args.max_length,
            template=args.template,
            precision=args.precision,
        )
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        return None
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    """Execute a CLI command and return a process exit code."""
    from opendecision.errors import OpenDecisionError

    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        if result is not None:
            _print(result)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, OpenDecisionError) as error:
        print(f"opendecision: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
