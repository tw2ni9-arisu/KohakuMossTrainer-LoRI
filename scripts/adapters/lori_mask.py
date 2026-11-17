"""Create compact zero-B LoRI initialization from raw calibration safetensors."""

import argparse
from pathlib import Path

from safetensors.torch import save_file

from kohakumoss.adapters.train.lori import make_sparse_init, read_lori_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sparsity", required=True, type=float)
    args = parser.parse_args()
    if args.calibration.resolve() == args.output.resolve():
        parser.error("Output must differ from calibration input")
    state, metadata = read_lori_state(args.calibration)
    sparse, mask_metadata = make_sparse_init(state, args.sparsity)
    save_file(sparse, args.output, metadata={**metadata, **mask_metadata})
    read_lori_state(args.output)
    print("Saved compact LoRI initialization")


if __name__ == "__main__":
    main()
