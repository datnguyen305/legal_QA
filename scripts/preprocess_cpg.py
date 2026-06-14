#!/usr/bin/env python3
"""Precompute and cache CPG curriculum records."""

from __future__ import annotations

import argparse

from data_preprocessing.cpg_preprocess import load_or_build_cpg_records


def parse_chunk_sizes(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="dataset/train_data.json")
    parser.add_argument("--dev-data", default="dataset/dev_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--cache-dir", default="cache/cpg")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--chunk-sizes", default="50,100,200,500")
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--easy-ratio", type=float, default=1.0)
    parser.add_argument("--easy-ratio-decay", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    chunk_sizes = parse_chunk_sizes(args.chunk_sizes)

    load_or_build_cpg_records(
        args.train_data,
        args.context_dir,
        args.train_limit,
        chunk_sizes,
        args.max_context_tokens,
        args.easy_ratio,
        seed=23,
        progress_label="initial train",
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    load_or_build_cpg_records(
        args.dev_data,
        args.context_dir,
        args.dev_limit,
        chunk_sizes,
        args.max_context_tokens,
        0.5,
        seed=23,
        progress_label="dev",
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    for epoch in range(1, args.epochs + 1):
        easy_ratio = max(0.0, args.easy_ratio - (epoch - 1) * args.easy_ratio_decay)
        load_or_build_cpg_records(
            args.train_data,
            args.context_dir,
            args.train_limit,
            chunk_sizes,
            args.max_context_tokens,
            easy_ratio,
            seed=23 + epoch,
            progress_label=f"train epoch {epoch}",
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
        )
    print(f"CPG preprocessing cache is ready in {args.cache_dir}")


if __name__ == "__main__":
    main()
