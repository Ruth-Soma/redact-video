"""Deterministic synthetic payment-screen video generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import OutputConfig
from redact_video.media.encoder import FrameEncoder
from redact_video.models import BoundingBox, SensitiveType

_MARKER_Y = 8
_MARKER_CELL_WIDTH = 10
_MARKER_HEIGHT = 12
_MARKER_BITS = 8


class ScenarioKind(str, Enum):
    STATIC_LANDSCAPE = "static_landscape"
    STATIC_PORTRAIT = "static_portrait"
    TYPING_PAN = "typing_pan"
    TYPING_CVV = "typing_cvv"
    SLOW_SCROLL = "slow_scroll"
    FAST_SCROLL = "fast_scroll"
    DISAPPEAR_RETURN = "disappear_return"
    MODAL_SLIDE = "modal_slide"
    SAVED_CARD = "saved_card"
    BRIEF_APPEARANCE = "brief_appearance"
    FALSE_POSITIVES = "false_positives"
    REPAIR_LEAK = "repair_leak"


@dataclass(frozen=True, slots=True)
class OracleToken:
    text: str
    confidence: float
    bbox: BoundingBox
    sensitive_type: SensitiveType | None = None


@dataclass(frozen=True, slots=True)
class GroundTruthRegion:
    type: SensitiveType
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class SyntheticFrameTruth:
    frame_index: int
    tokens: tuple[OracleToken, ...]
    sensitive_regions: tuple[GroundTruthRegion, ...]
    visible_regions: tuple[BoundingBox, ...]


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    kind: ScenarioKind
    path: Path
    width: int
    height: int
    frame_rate: float
    frames: tuple[SyntheticFrameTruth, ...]

    @property
    def duration_ms(self) -> int:
        return round(len(self.frames) / self.frame_rate * 1000)


def _draw_frame_marker(frame: NDArray[np.uint8], frame_index: int) -> None:
    if not 0 <= frame_index < 2**_MARKER_BITS:
        raise ValueError("Synthetic frame marker supports indices from 0 to 255")
    marker_x = frame.shape[1] - 8 - _MARKER_BITS * _MARKER_CELL_WIDTH
    for bit in range(_MARKER_BITS):
        value = 240 if frame_index & (1 << bit) else 15
        left = marker_x + bit * _MARKER_CELL_WIDTH
        cv2.rectangle(
            frame,
            (left, _MARKER_Y),
            (left + _MARKER_CELL_WIDTH - 1, _MARKER_Y + _MARKER_HEIGHT - 1),
            (value, value, value),
            -1,
        )


def decode_frame_marker(frame: NDArray[np.uint8]) -> int:
    """Decode the compression-resistant frame index embedded in a scenario."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Synthetic marker decoder expects a BGR frame")
    required_width = 8 + _MARKER_BITS * _MARKER_CELL_WIDTH
    required_height = _MARKER_Y + _MARKER_HEIGHT
    if frame.shape[1] < required_width or frame.shape[0] < required_height:
        raise ValueError("Frame is too small to contain a synthetic marker")
    marker_x = frame.shape[1] - 8 - _MARKER_BITS * _MARKER_CELL_WIDTH
    frame_index = 0
    for bit in range(_MARKER_BITS):
        left = marker_x + bit * _MARKER_CELL_WIDTH
        cell = frame[
            _MARKER_Y : _MARKER_Y + _MARKER_HEIGHT,
            left : left + _MARKER_CELL_WIDTH,
        ]
        if float(cell.mean()) >= 128:
            frame_index |= 1 << bit
    return frame_index


class _Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.frame = np.full((height, width, 3), (246, 247, 249), dtype=np.uint8)
        self.tokens: list[OracleToken] = []
        self.sensitive_regions: list[GroundTruthRegion] = []
        self.visible_regions: list[BoundingBox] = []
        cv2.rectangle(
            self.frame,
            (20, 34),
            (width - 21, height - 18),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            self.frame,
            (20, 34),
            (width - 21, height - 18),
            (215, 219, 225),
            1,
        )

    def text(
        self,
        value: str,
        x: int,
        y: int,
        *,
        scale: float = 0.58,
        thickness: int = 1,
        color: tuple[int, int, int] = (35, 38, 45),
        sensitive_type: SensitiveType | None = None,
        visible: bool = False,
    ) -> BoundingBox | None:
        if not value:
            return None
        (text_width, text_height), baseline = cv2.getTextSize(
            value,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        left = max(0, x)
        top = max(0, y)
        right = min(self.width, x + text_width + 5)
        bottom = min(self.height, y + text_height + baseline + 5)
        if right <= left or bottom <= top:
            return None
        clipped = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
        cv2.putText(
            self.frame,
            value,
            (x + 2, y + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        self.tokens.append(
            OracleToken(
                text=value,
                confidence=0.99,
                bbox=clipped,
                sensitive_type=sensitive_type,
            )
        )
        if sensitive_type is not None:
            self.sensitive_regions.append(
                GroundTruthRegion(type=sensitive_type, bbox=clipped)
            )
        if visible:
            self.visible_regions.append(clipped)
        return clipped

    def finish(self, frame_index: int) -> tuple[NDArray[np.uint8], SyntheticFrameTruth]:
        _draw_frame_marker(self.frame, frame_index)
        return self.frame, SyntheticFrameTruth(
            frame_index=frame_index,
            tokens=tuple(self.tokens),
            sensitive_regions=tuple(self.sensitive_regions),
            visible_regions=tuple(self.visible_regions),
        )


def _group_pan(digits: str) -> str:
    return " ".join(digits[index : index + 4] for index in range(0, len(digits), 4))


def _draw_payment_form(
    canvas: _Canvas,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    pan: str = "4242 4242 4242 4242",
    expiry: str = "08/29",
    cvv: str = "729",
) -> None:
    x = 42 + offset_x
    second_column = min(canvas.width - 135, 260 + offset_x)
    canvas.text("Payment Details", x, 48 + offset_y, scale=0.66, thickness=2)
    canvas.text("Card Number", x, 86 + offset_y)
    canvas.text(
        pan,
        x,
        116 + offset_y,
        scale=0.62,
        sensitive_type=SensitiveType.CARD_NUMBER,
    )
    canvas.text("Expiry", x, 174 + offset_y)
    canvas.text(
        expiry,
        x,
        204 + offset_y,
        scale=0.62,
        sensitive_type=SensitiveType.CARD_EXPIRY,
    )
    canvas.text("CVV", second_column, 174 + offset_y)
    canvas.text(
        cvv,
        second_column,
        204 + offset_y,
        scale=0.62,
        sensitive_type=SensitiveType.CARD_CVV,
    )
    canvas.text("Name on card", x, 258 + offset_y)
    canvas.text("Alex Example", x, 288 + offset_y, scale=0.62, visible=True)
    canvas.text("Billing address", x, 310 + offset_y)
    canvas.text("10 Example Street", x, 333 + offset_y, scale=0.54, visible=True)


def _draw_saved_card(canvas: _Canvas) -> None:
    canvas.text("Payment Methods", 42, 55, scale=0.66, thickness=2)
    canvas.text("Visa", 42, 112, scale=0.68, thickness=2, visible=True)
    canvas.text(
        "**** 4242",
        135,
        112,
        scale=0.68,
        sensitive_type=SensitiveType.SAVED_CARD_NUMBER,
    )
    canvas.text("Expires", 135, 166)
    canvas.text(
        "08/29",
        135,
        196,
        scale=0.62,
        sensitive_type=SensitiveType.SAVED_CARD_EXPIRY,
    )
    canvas.text("Use this card", 42, 270, scale=0.62, visible=True)


def _draw_false_positives(canvas: _Canvas) -> None:
    entries = (
        "OTP: 729",
        "Order 424242",
        "GBP 729.00",
        "Phone: +44 20 7946 0958",
        "Tracking number ZX424242",
        "Invoice ID 2029",
        "Postal code 729",
        "Date: 08/29",
        "Room 729",
    )
    canvas.text("Account Activity", 42, 50, scale=0.66, thickness=2, visible=True)
    for index, entry in enumerate(entries):
        canvas.text(entry, 42, 88 + index * 27, scale=0.52, visible=True)


class SyntheticCorpusGenerator:
    """Generate encoded scenarios and their per-frame semantic ground truth."""

    def __init__(
        self,
        root: Path,
        *,
        frame_rate: float = 10.0,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self.root = Path(root)
        self.frame_rate = frame_rate
        self.ffmpeg_binary = ffmpeg_binary

    def generate(self, kind: ScenarioKind) -> SyntheticScenario:
        width, height = self._dimensions(kind)
        frame_count = self._frame_count(kind)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{kind.value}.mp4"
        truths: list[SyntheticFrameTruth] = []
        with FrameEncoder(
            path,
            width=width,
            height=height,
            frame_rate=self.frame_rate,
            ffmpeg_binary=self.ffmpeg_binary,
            config=OutputConfig(preset="ultrafast", crf=12),
        ) as encoder:
            for frame_index in range(frame_count):
                frame, truth = self._frame(kind, frame_index, width, height)
                encoder.write(frame)
                truths.append(truth)
        return SyntheticScenario(
            kind=kind,
            path=path,
            width=width,
            height=height,
            frame_rate=self.frame_rate,
            frames=tuple(truths),
        )

    @staticmethod
    def _dimensions(kind: ScenarioKind) -> tuple[int, int]:
        if kind is ScenarioKind.STATIC_PORTRAIT:
            return 360, 640
        return 640, 360

    @staticmethod
    def _frame_count(kind: ScenarioKind) -> int:
        if kind is ScenarioKind.TYPING_PAN:
            return 18
        if kind is ScenarioKind.DISAPPEAR_RETURN:
            return 14
        return 12

    def _frame(
        self,
        kind: ScenarioKind,
        frame_index: int,
        width: int,
        height: int,
    ) -> tuple[NDArray[np.uint8], SyntheticFrameTruth]:
        canvas = _Canvas(width, height)
        if kind in {
            ScenarioKind.STATIC_LANDSCAPE,
            ScenarioKind.STATIC_PORTRAIT,
            ScenarioKind.REPAIR_LEAK,
        }:
            _draw_payment_form(canvas)
        elif kind is ScenarioKind.TYPING_PAN:
            digits = "4242424242424242"[: max(0, frame_index - 1)]
            _draw_payment_form(canvas, pan=_group_pan(digits), expiry="", cvv="")
        elif kind is ScenarioKind.TYPING_CVV:
            values = ("", "", "7", "72", "729", "729", "729", "729")
            cvv = values[min(frame_index, len(values) - 1)]
            _draw_payment_form(canvas, pan="", expiry="", cvv=cvv)
        elif kind is ScenarioKind.SLOW_SCROLL:
            _draw_payment_form(canvas, offset_y=-5 * frame_index)
        elif kind is ScenarioKind.FAST_SCROLL:
            _draw_payment_form(canvas, offset_y=-18 * frame_index)
        elif kind is ScenarioKind.DISAPPEAR_RETURN:
            if frame_index <= 4 or frame_index >= 9:
                _draw_payment_form(canvas)
            else:
                canvas.text("Order confirmed", 42, 320, scale=0.75, visible=True)
        elif kind is ScenarioKind.MODAL_SLIDE:
            offset_x = max(0, 210 - frame_index * 42)
            _draw_payment_form(canvas, offset_x=offset_x)
        elif kind is ScenarioKind.SAVED_CARD:
            _draw_saved_card(canvas)
        elif kind is ScenarioKind.BRIEF_APPEARANCE:
            if 4 <= frame_index <= 8:
                _draw_payment_form(canvas)
            else:
                canvas.text("Shopping basket", 42, 320, scale=0.75, visible=True)
        elif kind is ScenarioKind.FALSE_POSITIVES:
            _draw_false_positives(canvas)
        else:  # pragma: no cover - exhaustive enum safeguard
            raise ValueError(f"Unsupported synthetic scenario: {kind}")
        return canvas.finish(frame_index)
