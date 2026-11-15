"""Render original flight option text as OCR benchmark images (BVLM-6)."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bvlm.flight_data import PaperDataset, ROOT


def card_rows(data):
    for index, options in enumerate(data.heldout["all_options"]):
        yield {
            "split": "heldout", "set": index, "path": f"heldout/set-{index:03d}.png",
            "options": options,
        }
    for user in range(len(data.users)):
        for index, observed in enumerate(data.prompt_history(user, n_obs=5)):
            yield {
                "split": "interaction", "user": user, "round": index,
                "path": f"interaction/user-{user:03d}/round-{index}.png",
                "options": observed["options"],
            }


def render_ocr_card(row, font):
    from PIL import Image, ImageDraw

    margin, line_height, option_gap = 36, 39, 18
    # Keep the original wording, adding only line breaks for the image.
    blocks = [text.replace(", number of stops:", ",\nnumber of stops:") for text in row["options"]]
    lines = [line for block in blocks for line in block.splitlines()]
    width = max(font.getbbox(line)[2] for line in lines) + 2 * margin
    height = 2 * margin + len(lines) * line_height + (len(blocks) - 1) * option_gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for block in blocks:
        for line in block.splitlines():
            draw.text((margin, y), line, font=font, fill="black", anchor="lt")
            y += line_height
        y += option_gap
    return image


def write_card_images(data, output, render):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(prefix=".flight-cards-", dir=output.parent) as temporary:
        staged = Path(temporary) / "cards"
        staged.mkdir()
        with (staged / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for row in card_rows(data):
                target = staged / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                render(row).save(target, format="PNG")
                manifest.write(json.dumps(row) + "\n")
                count += 1
        staged.rename(output)
    return count


def write_cards(data, output):
    from PIL import ImageFont

    font = ImageFont.load_default(size=28)
    return write_card_images(data, output, lambda row: render_ocr_card(row, font))


def check_existing_cards(data, output):
    expected = list(card_rows(data))
    with (output / "manifest.jsonl").open(encoding="utf-8") as manifest:
        actual = [json.loads(line) for line in manifest]
    if actual != expected or any(not (output / row["path"]).is_file() for row in expected):
        raise ValueError(f"Incomplete or outdated card set: {output}. Choose a new folder with --out.")
    return len(expected)


def pillow_required(parser):
    # VS Code can start the system interpreter even after the repo venv was created.
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.is_file() and Path(sys.prefix).resolve() != environment.resolve():
        os.execv(str(python), [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    parser.exit(2, "Install Pillow first: python3 -m pip install -r requirements-ocr.txt\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Older commands used --all to request what is now the only mode.
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help="New directory for images and manifest")
    args = parser.parse_args()
    output = (args.out or ROOT / "data/generated/ocr").resolve()
    try:
        data = PaperDataset(ROOT / "benchmarks/text/data")
        if args.out is None and output.is_dir():
            count = check_existing_cards(data, output)
            print(f"All {count:,} PNG cards are already available in {output}")
        else:
            count = write_cards(data, output)
            print(f"{count:,} PNG cards written to {output}")
    except ImportError:
        pillow_required(parser)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot generate OCR cards: {error}\n")
    print(f"Test cards: {output / 'heldout'}")
    print(f"History cards: {output / 'interaction'}")
    print(f"Original texts and card paths: {output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
