"""Thin command-line interface for the shared redaction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from redact_video import __version__
from redact_video.config import AppConfig, ProcessingMode
from redact_video.config_loader import apply_processing_mode, load_app_config
from redact_video.exceptions import (
    ConfigurationError,
    RedactVideoError,
    UnsupportedInputError,
)
from redact_video.models import JobStatus
from redact_video.pipeline.automated import process_video_automated
from redact_video.pipeline.processor import default_output_path, process_video
from redact_video.redaction.mask import parse_manual_box
from redact_video.reporting import (
    SanitizedProcessReport,
    validate_report_destination,
    write_sanitized_report,
)

app = typer.Typer(
    name="redact-video",
    no_args_is_help=True,
    add_completion=False,
    help="Headless payment-card redaction for screen-recording videos.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"redact-video {__version__}")
        raise typer.Exit()


def _automatic_config(
    *, safe: bool, fast: bool, base_config: AppConfig | None = None
) -> AppConfig:
    if safe and fast:
        raise ConfigurationError("--safe and --fast cannot be used together")
    config = base_config or AppConfig()
    mode = (
        ProcessingMode.SAFE
        if safe
        else ProcessingMode.FAST
        if fast
        else ProcessingMode.STANDARD
    )
    return apply_processing_mode(config, mode)


def _format_timestamp(timestamp_ms: int) -> str:
    total_seconds, milliseconds = divmod(timestamp_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


@app.command()
def redact(
    input_path: Annotated[
        Path,
        typer.Argument(help="MP4 or MOV screen recording to process."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination MP4 path."),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Strict YAML engine configuration file.",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Write a sanitized JSON processing report.",
        ),
    ] = None,
    manual_box: Annotated[
        list[str] | None,
        typer.Option(
            "--manual-box",
            help="Static x,y,width,height region. Repeat for multiple regions.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing output file."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show pipeline progress."),
    ] = False,
    safe: Annotated[
        bool,
        typer.Option("--safe", help="Use the higher privacy-oriented OCR scan rate."),
    ] = False,
    fast: Annotated[
        bool,
        typer.Option("--fast", help="Use a lower OCR scan rate for development."),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = None,
) -> None:
    """Automatically redact card regions, or render explicit manual regions."""

    del version
    try:
        boxes = [parse_manual_box(value) for value in (manual_box or [])]
        destination = output or default_output_path(input_path)
        base_config = load_app_config(config_file) if config_file else AppConfig()
        if boxes:
            if safe or fast:
                raise ConfigurationError(
                    "--safe and --fast apply only to automatic mode"
                )
            if report is not None:
                raise ConfigurationError(
                    "--report is available only for automatic verified jobs"
                )
            result = process_video(
                input_path,
                output_path=destination,
                manual_boxes=boxes,
                config=base_config,
                overwrite=overwrite,
                progress=typer.echo if verbose else None,
            )
            automatic = False
        else:
            if report is not None:
                validate_report_destination(
                    report,
                    input_path=input_path,
                    output_path=destination,
                    overwrite=overwrite,
                )
            result = process_video_automated(
                input_path,
                output_path=destination,
                config=_automatic_config(
                    safe=safe,
                    fast=fast,
                    base_config=base_config,
                ),
                overwrite=overwrite,
                progress=typer.echo if verbose else None,
            )
            if report is not None:
                write_sanitized_report(
                    SanitizedProcessReport.from_result(result),
                    report,
                    overwrite=overwrite,
                )
            automatic = True
    except UnsupportedInputError as exc:
        typer.echo(f"Unsupported input: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ConfigurationError as exc:
        typer.echo(f"Invalid configuration: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    except RedactVideoError as exc:
        typer.echo(f"Processing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if automatic:
        typer.echo("Automatic detection/tracking render complete.")
        typer.echo(f"Masked frames: {result.masked_frames}")
        typer.echo(f"Mask regions: {result.mask_regions}")
        typer.echo(f"Verification passes: {result.verification.passes}")
        typer.echo(f"Repair windows: {result.verification.repairs}")
        typer.echo(f"Remaining possible leaks: {result.verification.remaining_leaks}")
        for issue in result.verification.potential_issues:
            typer.echo(
                "Potential issue: "
                f"{_format_timestamp(issue.start_ms)}–"
                f"{_format_timestamp(issue.end_ms)}"
            )
    else:
        typer.echo("Manual-mask render complete.")
    typer.echo(f"Status: {result.status.value.upper()}")
    typer.echo(f"Frames processed: {result.frames_processed}")
    typer.echo(f"Output: {result.output_path}")
    if report is not None:
        typer.echo(f"Report: {report}")
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    raise typer.Exit(code=0 if result.status is JobStatus.COMPLETED else 3)


if __name__ == "__main__":  # pragma: no cover
    app()
