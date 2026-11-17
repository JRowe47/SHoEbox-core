"""Train, inspect and sample the core model on local byte files."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train next-byte pairs from a local file")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--parameters", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--levels", type=int)
    train.add_argument("--nodes", type=int)
    train.add_argument("--facets", type=int)
    train.add_argument("--live-nodes", type=int)
    train.add_argument("--banks", type=int)
    train.add_argument("--experts", type=int)
    train.add_argument("--active-experts", type=int)
    train.add_argument("--offset", type=int, default=0, help="first input byte index in --data")
    train.add_argument("--targets", type=int, help="number of next-byte pairs; default: remaining file")
    train.add_argument("--chunk-size", type=int, default=65536)
    inspect = commands.add_parser("inspect", help="print model configuration and state")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    generate = commands.add_parser("generate", help="sample bytes from a checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--length", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=.8)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--out", type=Path, help="save exact bytes; otherwise display UTF-8")
    args = parser.parse_args()
    if args.command == "train":
        size = args.data.stat().st_size
        targets = size - 1 - args.offset if args.targets is None else args.targets
        if args.offset < 0 or targets <= 0 or args.offset + targets >= size or args.chunk_size <= 0:
            parser.error("require a valid offset, positive targets/chunk size, and a following target byte")
        creation = {key: getattr(args, key) for key in (
            "parameters", "seed", "levels", "nodes", "facets", "live_nodes",
            "banks", "experts", "active_experts") if getattr(args, key) is not None}
        if args.resume and creation:
            parser.error("--resume retains its saved topology and seed; omit creation options")
        data = np.memmap(args.data, dtype=np.uint8, mode="r")
        model = Model.load(args.resume) if args.resume else Model(**creation)
        with model:
            if not model.stats()["training"]:
                parser.error("--resume requires a trainable checkpoint")
            loss_sum = 0.
            end = args.offset + targets
            for start in range(args.offset, end, args.chunk_size):
                stop = min(end, start + args.chunk_size)
                loss_sum += model.run(data[start:stop], data[start+1:stop+1]) * (stop-start)
            model.save(args.out)
            print(json.dumps(dict(checkpoint=str(args.out), targets=targets,
                                  mean_nll=loss_sum/targets, next_offset=end,
                                  parameters=model.parameter_count), indent=2))
    else:
        with Model.load(args.checkpoint) as model:
            if args.command == "inspect":
                print(json.dumps(model.stats(), indent=2))
            else:
                output = model.generate(args.prompt, length=args.length,
                                        temperature=args.temperature, seed=args.seed)
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_bytes(output)
                else:
                    sys.stdout.write(output.decode("utf-8", "replace") + "\n")


if __name__ == "__main__":
    main()
