import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()

    numbers = np.fromfile(args.input_path, dtype="<u2")
    print(numbers)


if __name__ == "__main__":
    main()
