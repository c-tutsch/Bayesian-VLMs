"""Render the flight benchmark as diagrams with an optional distraction panel."""

import argparse
from functools import lru_cache, partial
from pathlib import Path
import random

from generate_ocr_cards import check_existing_cards, pillow_required, write_card_images
from bvlm.flight_data import Flight, PaperDataset, ROOT

BACKGROUND = "#F3F7FC"
INK = "#14253C"
MUTED = "#425771"
GRID = "#E1E8F2"
BORDER = "#D9E3EE"
FLIGHT_COLORS = ("#2657C7", "#087F6C", "#8747A5")
NOISE_LEFT = 1030

# A mark at the same position means the same value on every card.
PANELS = (
    ("departure_minutes", "Departure time", 36, 101, 1440,
     tuple((hour * 60, f"{hour:02d}:00") for hour in range(0, 25, 4))),
    ("duration_minutes", "Duration", 532, 101, 1200,
     tuple((hour * 60, f"{hour}h") for hour in range(0, 21, 5))),
    ("stops", "Stops", 36, 388, 2, ((0, "0"), (1, "1"), (2, "2"))),
    ("price", "Price (USD)", 532, 388, 1000,
     tuple((price, "$" + f"{price:,}") for price in range(0, 1001, 250))),
)
NOTICES = (
    ("Campus gallery", "Friday at 13:30", "Room 4"),
    ("Book exchange", "24 books available", "Shelf B"),
    ("Library desk", "Closes at 17:00", "Level 2"),
    ("Print workshop", "12 seats available", "Studio 5"),
    ("Club meeting", "Tuesday at 16:00", "Room 7"),
    ("Exhibition desk", "Programs available", "Counter 2"),
)
CHART_TITLES = ("Library visits", "Museum visitors", "Books returned", "Workshop sign-ups")


@lru_cache(maxsize=None)
def font(size):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def panel_x(origin, value, maximum):
    return origin + 111 + round(337 * value / maximum)


def diagram_template():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1300, 698), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((36, 27), "Flight options", font=font(36), fill=INK, stroke_width=1)
    draw.line((38, 81, 1262, 81), fill=BORDER, width=2)
    for _, title, x, y, maximum, ticks in PANELS:
        left, right, axis = x + 111, x + 448, y + 218
        draw.rounded_rectangle((x, y, x + 478, y + 269), radius=16,
                               fill="white", outline=BORDER, width=2)
        draw.text((x + 21, y + 19), title, font=font(24), fill=INK, stroke_width=1)
        for value, label in ticks:
            position = panel_x(x, value, maximum)
            draw.line((position, y + 64, position, axis + 4), fill=GRID, width=2)
            draw.text((position, axis + 12), label, font=font(16), fill=MUTED, anchor="mt")
        for index in range(3):
            row_y = y + 88 + index * 48
            draw.text((x + 21, row_y), f"Flight {index + 1}", font=font(19), fill=INK, anchor="lm")
            draw.line((left, row_y, right, row_y), fill="#EEF3F8", width=2)
        draw.line((left, axis, right, axis), fill="#9EACC0", width=2)
    return image


def diagram_fields(data, row):
    if row["split"] == "heldout":
        options = data.heldout["all_options_numpy"][row["set"]]
    else:
        options = data.users[row["user"]]["rounds_numpy"][row["round"]]

    fields = []
    for values in options:
        flight = Flight.from_values(values).fields
        hours, minutes = map(int, flight["departure"].split(":"))
        flight["departure_minutes"] = 60 * hours + minutes
        fields.append(flight)
    return fields


def draw_distractions(draw, relative_path):
    # The card path seeds the draw independently of the values and user choice.
    randomizer = random.Random(f"flight-card-noise-v1:{relative_path}")
    left, top = NOISE_LEFT, 101
    axis = top + 183
    draw.rounded_rectangle((left, top, 1263, top + 234), radius=14,
                           fill="#F0ECFA", outline=BORDER, width=2)
    draw.text((left + 17, top + 18), randomizer.choice(CHART_TITLES),
              font=font(21), fill="#52397C")
    for value in (0, 50, 100):
        position = axis - round(value * 1.15)
        draw.line((left + 42, position, left + 211, position), fill="#DFD6F0", width=2)
        draw.text((left + 34, position), str(value), font=font(14), fill=MUTED, anchor="rm")
    for index, day in enumerate(("Mon", "Wed", "Fri")):
        center = left + 65 + index * 56
        height = round(randomizer.randint(15, 95) * 1.15)
        draw.rounded_rectangle((center - 15, axis - height, center + 15, axis),
                               radius=4, fill="#8053A9")
        draw.text((center, axis + 10), day, font=font(15), fill=MUTED, anchor="mt")
    for index, (title, first, second) in enumerate(randomizer.sample(NOTICES, 2)):
        y = 353 + index * 132
        fill, accent = (("#FFF4D7", "#775009"), ("#E9F1FD", "#284983"))[index]
        draw.rounded_rectangle((left, y, 1263, y + 113), radius=14,
                               fill=fill, outline=BORDER, width=2)
        draw.text((left + 17, y + 14), title, font=font(21), fill=accent)
        draw.text((left + 17, y + 49), first, font=font(18), fill=INK)
        draw.text((left + 17, y + 75), second, font=font(18), fill=INK)


def render_diagram(row, *, data, template, with_noise):
    from PIL import ImageDraw

    image = template.copy()
    draw = ImageDraw.Draw(image)
    fields = diagram_fields(data, row)
    for field, _, x, y, maximum, _ in PANELS:
        for index, (flight, color) in enumerate(zip(fields, FLIGHT_COLORS)):
            row_y = y + 88 + index * 48
            position = panel_x(x, flight[field], maximum)
            if field in ("price", "duration_minutes"):
                draw.rounded_rectangle((x + 111, row_y - 8, position, row_y + 8),
                                       radius=5, fill=color)
            else:
                draw.ellipse((position - 9, row_y - 9, position + 9, row_y + 9),
                             fill=color, outline="white", width=2)
    if with_noise:
        draw_distractions(draw, row["path"])
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/generated",
                        help="Parent directory for the two diagram card sets")
    args = parser.parse_args()
    try:
        data = PaperDataset(ROOT / "benchmarks/text/data")
        template = diagram_template()
        for name, with_noise in (("diagrams", False), ("diagrams-noise", True)):
            output = (args.out / name).resolve()
            if output.is_dir():
                count = check_existing_cards(data, output)
                print(f"All {count:,} PNG cards are already available in {output}", flush=True)
            else:
                draw = partial(render_diagram, data=data, template=template, with_noise=with_noise)
                count = write_card_images(data, output, draw)
                print(f"{count:,} PNG cards written to {output}", flush=True)
    except ImportError:
        pillow_required(parser)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot generate diagram cards: {error}\n")


if __name__ == "__main__":
    main()
