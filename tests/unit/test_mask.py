import pytest

from redact_video.exceptions import ConfigurationError
from redact_video.models import BoundingBox
from redact_video.redaction.mask import parse_manual_box


def test_parse_manual_box() -> None:
    assert parse_manual_box("10, 20, 30, 40") == BoundingBox(
        x=10, y=20, width=30, height=40
    )


@pytest.mark.parametrize(
    "value",
    ["1,2,3", "1,2,3,4,5", "a,2,3,4", "1,2,0,4", "-1,2,3,4"],
)
def test_parse_manual_box_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ConfigurationError):
        parse_manual_box(value)
