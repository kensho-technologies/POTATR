# Copyright 2025-present Kensho Technologies, LLC.
r"""Make the checkpoint that POTATR training starts from, from the public TATR-v1.1-Pub weights.

TATR-v1.1-Pub is Table Transformer v1.1 trained on PubTables-1M (MIT license), available at
https://huggingface.co/bsmock/TATR-v1.1-Pub. POTATR has the same backbone, transformer and box
head, so those tensors are copied unchanged. TATR has 125 object queries; POTATR's extra queries
start from random values drawn with a fixed seed, so the checkpoint is the same every time. TATR's
class head is left out, since POTATR doesn't have one; POTATR's node and edge heads start from
random values when training begins.

Example:
    potatr-init \
        --config_file configs/model/potatr_v1.json \
        --tatr_weights TATR-v1.1-Pub-msft.pth \
        --output potatr_init.pth
"""

import argparse
import hashlib
import logging

import torch

from potatr.config import load_config_files


logger = logging.getLogger(__name__)

# SHA-256 of TATR-v1.1-Pub-msft.pth as released at https://huggingface.co/bsmock/TATR-v1.1-Pub.
TATR_V1_1_PUB_SHA256 = "8147ced2c4a3045ae90b7dca3a2164208b80de70739275a9592b2c88fddc04ae"
SEED = 42


def file_sha256(path: str) -> str:
    """Return the SHA-256 of a file, as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_initial_checkpoint(
    tatr_state_dict: dict[str, torch.Tensor], num_queries: int, seed: int = SEED
) -> dict[str, torch.Tensor]:
    """Return POTATR's starting weights, made from a Table Transformer state dict.

    Every tensor except the class head (class_embed) is copied. The query embeddings keep TATR's
    queries as the first rows and add num_queries minus that many rows drawn from a standard
    normal distribution (the initialization of torch.nn.Embedding) with the given seed.
    """
    queries = tatr_state_dict["query_embed.weight"]
    if num_queries < queries.shape[0]:
        raise ValueError(
            f"num_queries is {num_queries}, but the TATR weights have {queries.shape[0]} queries"
        )
    generator = torch.Generator().manual_seed(seed)
    extra_queries = torch.randn(
        num_queries - queries.shape[0], queries.shape[1], generator=generator, dtype=queries.dtype
    )
    state_dict = {
        name: tensor
        for name, tensor in tatr_state_dict.items()
        if not name.startswith("class_embed.")
    }
    state_dict["query_embed.weight"] = torch.cat([queries, extra_queries])
    return state_dict


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Make POTATR's starting checkpoint from the TATR-v1.1-Pub weights"
    )
    parser.add_argument("--config_file", required=True, nargs="+", help="The model config file(s)")
    parser.add_argument(
        "--tatr_weights", required=True, help="TATR-v1.1-Pub-msft.pth, from Hugging Face"
    )
    parser.add_argument("--output", required=True, help="Path to write the checkpoint to")
    return parser.parse_args()


def cli_main() -> None:
    """CLI entry point for making the starting checkpoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = get_args()
    sha256 = file_sha256(args.tatr_weights)
    if sha256 != TATR_V1_1_PUB_SHA256:
        raise ValueError(
            f"{args.tatr_weights} is not the released TATR-v1.1-Pub-msft.pth "
            f"(SHA-256 {sha256}, expected {TATR_V1_1_PUB_SHA256})"
        )
    config = load_config_files(args.config_file)
    tatr_state_dict = torch.load(args.tatr_weights, map_location="cpu")
    if tatr_state_dict["query_embed.weight"].shape[1] != config["hidden_dim"]:
        raise ValueError(
            f"hidden_dim is {config['hidden_dim']}, but the TATR weights have "
            f"{tatr_state_dict['query_embed.weight'].shape[1]}"
        )
    state_dict = make_initial_checkpoint(tatr_state_dict, config["num_queries"])
    torch.save(state_dict, args.output)
    logger.info(
        "Wrote %d tensors to %s (%d queries, %d of them new, seed %d)",
        len(state_dict),
        args.output,
        config["num_queries"],
        config["num_queries"] - tatr_state_dict["query_embed.weight"].shape[0],
        SEED,
    )


if __name__ == "__main__":
    cli_main()
