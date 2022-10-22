# Script originally by Harubaru: https://github.com/harubaru/

import os
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path to a model file",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        required=False,
        help="Overwrite the original checkpoint with the pruned version",
    )
    args = parser.parse_args()
    return args


def prune(p, overwrite=False):
    size_initial = os.path.getsize(p)
    nsd = dict()
    sd = torch.load(p, map_location="cpu")
    for k in sd.keys():
        if k != "optimizer_states":
            nsd[k] = sd[k]
        sd = nsd['state_dict'].copy()
        new_sd = dict()
        for k in sd:
            new_sd[k] = sd[k].half()
        nsd['state_dict'] = new_sd

    fn = f"{os.path.splitext(p)[0]}-pruned.ckpt"
    if overwrite:
        fn = p

    print(f"Saving pruned checkpoint at: {fn}")
    torch.save(nsd, fn)

if __name__ == "__main__":
    args = parse_args()
    prune(args.model, args.overwrite)
