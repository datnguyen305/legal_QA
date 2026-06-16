#!/usr/bin/env python3
"""Build cached extractive records for QANet/Cross-Passage/Deep Cascade/TD-SAN."""

from __future__ import annotations

import argparse

from train_extractive import load_or_build_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--cache-dir", default="cache/extractive")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-passages", type=int, default=6)
    parser.add_argument("--passage-len", type=int, default=256)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-disk-cache", action="store_true")
    args = parser.parse_args()

    use_cache = not args.no_disk_cache
    train_rows = load_or_build_records(
        args.train_data,
        args.context_dir,
        args.train_limit,
        args.max_context_chars,
        args.max_passages,
        args.passage_len,
        "train",
        args.cache_dir,
        use_cache,
        args.rebuild_cache,
    )
    dev_rows = load_or_build_records(
        args.dev_data,
        args.context_dir,
        args.dev_limit,
        args.max_context_chars,
        args.max_passages,
        args.passage_len,
        "dev",
        args.cache_dir,
        use_cache,
        args.rebuild_cache,
    )
    print(f"Cached extractive records: train={len(train_rows)} dev={len(dev_rows)} cache_dir={args.cache_dir}")


if __name__ == "__main__":
    main()
