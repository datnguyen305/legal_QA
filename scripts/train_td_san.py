#!/usr/bin/env python3
"""Train TD-SAN / DynSAN reader."""

from __future__ import annotations

import sys

from train_extractive import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "td_san"]
    main()
