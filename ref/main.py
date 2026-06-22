import sys
from collections import Counter
from pathlib import Path

import regex


GPT2_REGEX = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
)
SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> Path:
    if len(sys.argv) != 2:
        raise ValueError(f"usage: {sys.argv[0]} <file_path>")

    file_path = Path(sys.argv[1]).resolve(strict=True)

    return file_path


def main() -> None:
    file_path = parse_args()

    content = file_path.read_bytes()
    text = content.decode("utf-8")

    freq_map: Counter[bytes] = Counter()
    for piece in text.split(SPECIAL_TOKEN):
        for match in GPT2_REGEX.finditer(piece):
            freq_map[match.group(0).encode("utf-8")] += 1

    print(f"result: {len(freq_map)} entries")
    print(f"result: {sum(freq_map.values())} total tokens")


if __name__ == "__main__":
    main()
