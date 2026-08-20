"""Local Luhn validation with no logging or dependencies."""

from __future__ import annotations


def luhn_valid(number: str) -> bool:
    """Validate a 13-19 digit PAN candidate, accepting spaces and hyphens."""

    compact = number.replace(" ", "").replace("-", "")
    if not compact.isascii() or not compact.isdigit() or not 13 <= len(compact) <= 19:
        return False

    total = 0
    parity = len(compact) % 2
    for index, character in enumerate(compact):
        digit = ord(character) - ord("0")
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
