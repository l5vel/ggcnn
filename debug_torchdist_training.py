import torch
import torch.distributed as dist
import os
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"Starting main() with local_rank: {args.local_rank}")
    logging.info(f"All Environment Variables: {os.environ}")
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        logging.info(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    except RuntimeError as e:
        logging.error(f"Initialization error: {e}")
    logging.info(f"sys.argv: {sys.argv}")

if __name__ == "__main__":
    main()