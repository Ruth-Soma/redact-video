"""Opt-in acceptance runner for external, non-production recordings."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from redact_video.config import AppConfig, ProcessingMode
from redact_video.config_loader import apply_processing_mode, load_app_config
from redact_video.exceptions import (
    ConfigurationError,
    RedactVideoError,
    UnsupportedInputError,
)
from redact_video.models import JobStatus
from redact_video.pipeline.automated import (
    AutomatedProcessResult,
    process_video_automated,
)
from redact_video.reporting import (
    SanitizedResultReport,
    write_private_json_artifact,
)

AcceptanceProcessor = Callable[..., AutomatedProcessResult]


class AcceptanceSample(BaseModel):
    """One value-free sample result with no source or output filename."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(pattern=r"^sample_[0-9]{3}$")
    status: JobStatus
    elapsed_ms: int = Field(ge=0)
    realtime_factor: float | None = Field(default=None, ge=0)
    performance_passed: bool
    result: SanitizedResultReport | None = None
    error_code: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> AcceptanceSample:
        if self.status is JobStatus.FAILED:
            if self.error_code is None or self.result is not None:
                raise ValueError("Failed samples require only a sanitized error code")
        elif self.result is None or self.error_code is not None:
            raise ValueError("Processed samples require only a sanitized result")
        return self


class AcceptanceReport(BaseModel):
    """Aggregate report safe to retain alongside reviewed outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    created_at: datetime
    mode: ProcessingMode
    samples: tuple[AcceptanceSample, ...]
    total_samples: int = Field(ge=1)
    completed: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    failed: int = Field(ge=0)
    performance_failures: int = Field(ge=0)
    acceptance_passed: bool


def _discover_recordings(input_dir: Path, *, maximum_samples: int) -> tuple[Path, ...]:
    source_dir = Path(input_dir)
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ConfigurationError("Acceptance input must be an existing directory")
    recordings = tuple(
        path
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name.casefold())
        if path.suffix.lower() in {".mp4", ".mov"}
        and path.is_file()
        and not path.is_symlink()
    )
    if not recordings:
        raise ConfigurationError("Acceptance input contains no MP4 or MOV recordings")
    if len(recordings) > maximum_samples:
        raise ConfigurationError("Acceptance input exceeds the configured sample limit")
    return recordings


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, UnsupportedInputError):
        return "unsupported_input"
    if isinstance(exc, RedactVideoError):
        return "processing_failed"
    return "internal_error"


def run_acceptance(
    input_dir: Path,
    output_dir: Path,
    *,
    report_path: Path,
    config: AppConfig,
    mode: ProcessingMode = ProcessingMode.SAFE,
    overwrite: bool = False,
    maximum_samples: int = 50,
    maximum_realtime_factor: float | None = None,
    processor: AcceptanceProcessor | None = None,
    progress: Callable[[str], None] | None = None,
) -> AcceptanceReport:
    """Process a private external corpus without persisting source identifiers."""

    if maximum_samples < 1:
        raise ConfigurationError("Acceptance sample limit must be positive")
    if maximum_realtime_factor is not None and maximum_realtime_factor <= 0:
        raise ConfigurationError("Maximum realtime factor must be positive")
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    try:
        same_directory = source_dir.resolve() == destination_dir.resolve()
    except OSError:
        same_directory = source_dir.absolute() == destination_dir.absolute()
    if same_directory:
        raise ConfigurationError(
            "Acceptance output must differ from the input directory"
        )
    recordings = _discover_recordings(source_dir, maximum_samples=maximum_samples)
    report_destination = Path(report_path)
    if report_destination.suffix.lower() != ".json":
        raise ConfigurationError("Acceptance report must use the .json extension")
    if report_destination.exists() and not overwrite:
        raise ConfigurationError(
            "Acceptance report already exists; use --overwrite to replace it"
        )
    try:
        destination_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise ConfigurationError(
            "Acceptance output directory could not be created"
        ) from None
    if not destination_dir.is_dir() or destination_dir.is_symlink():
        raise ConfigurationError("Acceptance output must be a regular directory")
    active_config = apply_processing_mode(config, mode)
    active_processor = processor or process_video_automated

    output_paths = tuple(
        destination_dir / f"sample_{index:03d}.redacted.mp4"
        for index in range(1, len(recordings) + 1)
    )
    if not overwrite and any(path.exists() for path in output_paths):
        raise ConfigurationError(
            "Acceptance output already exists; use --overwrite to replace it"
        )

    samples: list[AcceptanceSample] = []
    for index, (source, destination) in enumerate(
        zip(recordings, output_paths, strict=True),
        start=1,
    ):
        sample_id = f"sample_{index:03d}"
        if progress:
            progress(f"{sample_id}: processing")
        started = time.perf_counter()
        try:
            result = active_processor(
                source,
                output_path=destination,
                config=active_config,
                overwrite=overwrite,
            )
            elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
            realtime_factor = elapsed_ms / result.input_duration_ms
            performance_passed = (
                maximum_realtime_factor is None
                or realtime_factor <= maximum_realtime_factor
            )
            samples.append(
                AcceptanceSample(
                    sample_id=sample_id,
                    status=result.status,
                    elapsed_ms=elapsed_ms,
                    realtime_factor=realtime_factor,
                    performance_passed=performance_passed,
                    result=SanitizedResultReport.from_result(result),
                )
            )
        except Exception as exc:  # noqa: BLE001 -- sanitize every corpus failure
            elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
            samples.append(
                AcceptanceSample(
                    sample_id=sample_id,
                    status=JobStatus.FAILED,
                    elapsed_ms=elapsed_ms,
                    performance_passed=False,
                    error_code=_failure_code(exc),
                )
            )

    completed = sum(sample.status is JobStatus.COMPLETED for sample in samples)
    needs_review = sum(sample.status is JobStatus.NEEDS_REVIEW for sample in samples)
    failed = sum(sample.status is JobStatus.FAILED for sample in samples)
    performance_failures = sum(
        sample.status is not JobStatus.FAILED and not sample.performance_passed
        for sample in samples
    )
    report = AcceptanceReport(
        created_at=datetime.now(UTC),
        mode=mode,
        samples=tuple(samples),
        total_samples=len(samples),
        completed=completed,
        needs_review=needs_review,
        failed=failed,
        performance_failures=performance_failures,
        acceptance_passed=(
            completed == len(samples)
            and needs_review == 0
            and failed == 0
            and performance_failures == 0
        ),
    )
    write_private_json_artifact(report, report_path, overwrite=overwrite)
    return report


app = typer.Typer(
    name="redact-video-acceptance",
    no_args_is_help=True,
    add_completion=False,
    help="Run private, external V1 acceptance recordings sequentially.",
)


@app.command()
def acceptance(
    input_dir: Annotated[Path, typer.Argument(help="Private MP4/MOV corpus.")],
    output_dir: Annotated[Path, typer.Argument(help="Reviewed redacted outputs.")],
    report: Annotated[
        Path,
        typer.Option("--report", help="Sanitized aggregate JSON report."),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Strict YAML engine configuration."),
    ] = None,
    mode: Annotated[
        ProcessingMode,
        typer.Option("--mode", help="Processing scan-rate profile."),
    ] = ProcessingMode.SAFE,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace sample outputs and report."),
    ] = False,
    maximum_samples: Annotated[
        int,
        typer.Option("--max-samples", min=1, max=1000),
    ] = 50,
    maximum_realtime_factor: Annotated[
        float | None,
        typer.Option("--max-realtime-factor", min=0.01),
    ] = None,
) -> None:
    try:
        base_config = load_app_config(config_file) if config_file else AppConfig()
        result = run_acceptance(
            input_dir,
            output_dir,
            report_path=report,
            config=base_config,
            mode=mode,
            overwrite=overwrite,
            maximum_samples=maximum_samples,
            maximum_realtime_factor=maximum_realtime_factor,
            progress=typer.echo,
        )
    except ConfigurationError as exc:
        typer.echo(f"Invalid acceptance configuration: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    except RedactVideoError as exc:
        typer.echo(f"Acceptance processing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Completed: {result.completed}/{result.total_samples}")
    typer.echo(f"Needs review: {result.needs_review}")
    typer.echo(f"Failed: {result.failed}")
    typer.echo(f"Performance failures: {result.performance_failures}")
    raise typer.Exit(
        code=0
        if result.acceptance_passed
        else 1
        if result.failed or result.performance_failures
        else 3
    )
