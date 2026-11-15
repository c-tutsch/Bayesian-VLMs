"""Render the four-feature flight options from Zenodo record 17677329."""

import re
from typing import Sequence


_OPTION = re.compile(
    r"Flight (?P<label>[1-3]): departure time: "
    r"(?P<hour>0[1-9]|1[0-2]):(?P<minute>[0-5][0-9]) (?P<period>AM|PM), "
    r"duration: (?:(?P<hours>[0-9]+) hr(?: (?P<minutes>[0-9]+) min)?"
    r"|(?P<minutes_only>[0-9]+) min), "
    r"number of stops: (?P<stops>[0-9]+), price: \$(?P<price>[0-9]+)"
)


def render_flight_card(options: Sequence[str]) -> str:
    """Format three source options; preserve labels/order, units and dollar sign.

    This custom card uses a 24-hour clock; source strings remain unchanged.
    Pass only option strings, never records containing preferences or labels.
    """
    if isinstance(options, (str, bytes)) or len(options) != 3:
        raise ValueError("A flight card requires exactly three option strings.")

    lines = ["Flight options"]
    for number, option in enumerate(options, start=1):
        match = _OPTION.fullmatch(option) if isinstance(option, str) else None
        if match is None or int(match['label']) != number:
            raise ValueError(
                f"Option {number} must use the original four-feature format "
                f"and the label Flight {number}."
            )

        hour = int(match['hour']) % 12 + (12 if match['period'] == 'PM' else 0)
        total_minutes = (
            int(match['hours'] or 0) * 60
            + int(match['minutes'] or match['minutes_only'] or 0)
        )
        hours, minutes = divmod(total_minutes, 60)
        lines.append(
            f"Flight {number}: Departure: {hour:02d}:{match['minute']} (24-hour) | "
            f"Duration: {hours} h {minutes} min | Stops: {int(match['stops'])} | "
            f"Price: ${match['price']}"
        )
    return "\n".join(lines)
