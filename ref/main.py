import sys
from collections import Counter
from pathlib import Path

import regex


GPT2_REGEX = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
)
SPECIAL_TOKEN = "<|endoftext|>"


def parse_args() -> tuple[Path, Path | None]:
    if len(sys.argv) not in (2, 4):
        raise ValueError(f"usage: {sys.argv[0]} <file_path> [-i input.tsv]")

    file_path = Path(sys.argv[1]).resolve(strict=True)

    input_path = None
    if len(sys.argv) == 4:
        if sys.argv[2] != "-i":
            raise ValueError(f"usage: {sys.argv[0]} <file_path> [-i input.tsv]")
        input_path = Path(sys.argv[3]).resolve(strict=True)

    return file_path, input_path


def read_freq_map(path: Path) -> dict[bytes, int]:
    freq_map: dict[bytes, int] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                hex_key, count = line.split("\t", maxsplit=1)
                freq_map[bytes.fromhex(hex_key)] = int(count)
            except ValueError as err:
                raise ValueError(f"invalid TSV line {line_number}: {line!r}") from err

    return freq_map


def compare_freq_maps(rust_map: dict[bytes, int], python_map: Counter[bytes]) -> None:
    if rust_map == python_map:
        print("compare: match")
        return

    rust_keys = set(rust_map)
    python_keys = set(python_map)
    rust_only = sorted(rust_keys - python_keys)
    python_only = sorted(python_keys - rust_keys)
    different_counts = sorted(
        key for key in rust_keys & python_keys if rust_map[key] != python_map[key]
    )

    print("compare: mismatch")
    print(f"rust-only keys: {len(rust_only)}")
    print(f"python-only keys: {len(python_only)}")
    print(f"different counts: {len(different_counts)}")

    for key in rust_only[:5]:
        print(f"example rust-only: {key.hex()} rust={rust_map[key]}")

    for key in python_only[:5]:
        print(f"example python-only: {key.hex()} python={python_map[key]}")

    for key in different_counts[:5]:
        print(f"example different: {key.hex()} rust={rust_map[key]} python={python_map[key]}")


def main() -> None:
    file_path, input_path = parse_args()

    content = file_path.read_bytes()
    text = content.decode("utf-8")

    freq_map: Counter[bytes] = Counter()
    for piece in text.split(SPECIAL_TOKEN):
        for match in GPT2_REGEX.finditer(piece):
            freq_map[match.group(0).encode("utf-8")] += 1

    print(f"result: {len(freq_map)} entries")
    print(f"result: {sum(freq_map.values())} total tokens")

    if input_path is not None:
        rust_map = read_freq_map(input_path)
        compare_freq_maps(rust_map, freq_map)


if __name__ == "__main__":
    main()
