# Credit to Victarry https://github.com/Victarry/stable-dreambooth for the original creation of this script

from diffusers.pipelines import StableDiffusionPipeline
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = "cuda"
    model = StableDiffusionPipeline.from_pretrained(args.model_path).to(device)
    tokenizer = model.tokenizer

    rare_tokens = []
    for k, v in tokenizer.encoder.items():
        if len(k) <= 3 and 40000 > v > 35000:
            rare_tokens.append(k)
    
    identifiers = []
    for _ in range(3):
        idx = random.randint(0, len(rare_tokens))
        identifiers.append(rare_tokens[idx])
    
    print(" ".join(identifiers))
