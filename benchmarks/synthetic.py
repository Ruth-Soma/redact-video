"""Repeatable end-to-end performance benchmark over the synthetic corpus."""

from __future__ import annotations

import argparse
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import AppConfig, OutputConfig
from redact_video.models import JobStatus, SensitiveType
from redact_video.pipeline.automated import process_video_automated
from redact_video.reporting import write_private_json_artifact
from tests.synthetic.generator import (
    ScenarioKind,
    SyntheticCorpusGenerator,
    SyntheticScenario,
)
from tests.synthetic.metrics import measure_mask_timeline
from tests.synthetic.oracle import ScenarioOcrEngine

DEFAULT_SCENARIOS = (
    ScenarioKind.STATIC_LANDSCAPE,
    ScenarioKind.TYPING_PAN,
    ScenarioKind.FAST_SCROLL,
    ScenarioKind.REPAIR_LEAK,
)


class ScenarioBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: ScenarioKind
    status: JobStatus
    frames: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    elapsed_ms: int = Field(ge=0)
    throughput_fps: float = Field(ge=0)
    realtime_factor: float = Field(ge=0)
    verification_passes: int = Field(ge=0)
    repairs: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    missed_sensitive_pixels: int = Field(ge=0)
    overmask_ratio: float = Field(ge=0)
    masked_visible_pixels: int = Field(ge=0)
    privacy_passed: bool
    performance_passed: bool


class SyntheticBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    created_at: datetime
    python_version: str
    operating_system: str
    architecture: str
    scenarios: tuple[ScenarioBenchmark, ...]
    total_elapsed_ms: int = Field(ge=0)
    privacy_failures: int = Field(ge=0)
    performance_failures: int = Field(ge=0)
    benchmark_passed: bool


def _ocr_engines(
    scenario: SyntheticScenario,
) -> tuple[ScenarioOcrEngine, ScenarioOcrEngine]:
    if scenario.kind is ScenarioKind.REPAIR_LEAK:
        return (
            ScenarioOcrEngine(
                scenario,
                excluded_types=frozenset({SensitiveType.CARD_CVV}),
                excluded_texts=frozenset({"CVV"}),
            ),
            ScenarioOcrEngine(
                scenario,
                excluded_types=frozenset(
                    {SensitiveType.CARD_NUMBER, SensitiveType.CARD_EXPIRY}
                ),
                sensitive_passes=1,
            ),
        )
    return (
        ScenarioOcrEngine(scenario),
        ScenarioOcrEngine(scenario, include_sensitive=False),
    )


def run_synthetic_benchmark(
    kinds: tuple[ScenarioKind, ...],
    *,
    report_path: Path,
    overwrite: bool = False,
    maximum_realtime_factor: float | None = None,
) -> SyntheticBenchmarkReport:
    if not kinds:
        raise ValueError("At least one benchmark scenario is required")
    if maximum_realtime_factor is not None and maximum_realtime_factor <= 0:
        raise ValueError("Maximum realtime factor must be positive")
    started_all = time.perf_counter()
    scenario_results: list[ScenarioBenchmark] = []

    with tempfile.TemporaryDirectory(prefix="video-redactor-benchmark.") as temp:
        root = Path(temp)
        generator = SyntheticCorpusGenerator(root / "corpus")
        for kind in kinds:
            scenario = generator.generate(kind)
            output_path = root / f"{kind.value}.redacted.mp4"
            source_ocr, verification_ocr = _ocr_engines(scenario)
            started = time.perf_counter()
            result = process_video_automated(
                scenario.path,
                output_path=output_path,
                config=AppConfig(output=OutputConfig(preset="ultrafast")),
                ocr_engine=source_ocr,
                verification_ocr_engine=verification_ocr,
            )
            elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
            duration_ms = scenario.duration_ms
            metrics = measure_mask_timeline(scenario, result.mask_timeline)
            realtime_factor = elapsed_ms / duration_ms
            throughput_fps = (
                len(scenario.frames) * 1000 / elapsed_ms
                if elapsed_ms
                else float(len(scenario.frames) * 1000)
            )
            privacy_passed = (
                result.status is JobStatus.COMPLETED
                and metrics.missed_sensitive_pixels == 0
                and metrics.masked_visible_pixels == 0
            )
            performance_passed = (
                maximum_realtime_factor is None
                or realtime_factor <= maximum_realtime_factor
            )
            scenario_results.append(
                ScenarioBenchmark(
                    scenario=kind,
                    status=result.status,
                    frames=len(scenario.frames),
                    duration_ms=duration_ms,
                    elapsed_ms=elapsed_ms,
                    throughput_fps=throughput_fps,
                    realtime_factor=realtime_factor,
                    verification_passes=result.verification.passes,
                    repairs=result.verification.repairs,
                    coverage=metrics.coverage,
                    missed_sensitive_pixels=metrics.missed_sensitive_pixels,
                    overmask_ratio=metrics.overmask_ratio,
                    masked_visible_pixels=metrics.masked_visible_pixels,
                    privacy_passed=privacy_passed,
                    performance_passed=performance_passed,
                )
            )

    privacy_failures = sum(not item.privacy_passed for item in scenario_results)
    performance_failures = sum(not item.performance_passed for item in scenario_results)
    report = SyntheticBenchmarkReport(
        created_at=datetime.now(UTC),
        python_version=platform.python_version(),
        operating_system=platform.system().lower(),
        architecture=platform.machine().lower(),
        scenarios=tuple(scenario_results),
        total_elapsed_ms=max(0, round((time.perf_counter() - started_all) * 1000)),
        privacy_failures=privacy_failures,
        performance_failures=performance_failures,
        benchmark_passed=privacy_failures == 0 and performance_failures == 0,
    )
    write_private_json_artifact(report, report_path, overwrite=overwrite)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete pipeline on generated videos."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[kind.value for kind in ScenarioKind],
        help="Scenario to run; repeat for more than one.",
    )
    parser.add_argument("--all", action="store_true", help="Run every scenario.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic-benchmark.json"),
        help="Sanitized JSON result path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-realtime-factor", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.all and args.scenario:
        raise SystemExit("--all cannot be combined with --scenario")
    if args.all:
        kinds = tuple(ScenarioKind)
    elif args.scenario:
        kinds = tuple(ScenarioKind(value) for value in args.scenario)
    else:
        kinds = DEFAULT_SCENARIOS
    report = run_synthetic_benchmark(
        kinds,
        report_path=args.output,
        overwrite=args.overwrite,
        maximum_realtime_factor=args.max_realtime_factor,
    )
    for result in report.scenarios:
        print(
            f"{result.scenario.value}: {result.throughput_fps:.2f} fps, "
            f"{result.realtime_factor:.2f}x realtime, "
            f"privacy={'pass' if result.privacy_passed else 'fail'}"
        )
    print(f"Report: {args.output}")
    return 0 if report.benchmark_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
