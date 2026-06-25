#!/usr/bin/env python3
from __future__ import annotations

import sys
from train_abstractive_baseline import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "dcmn_plus"]
    main()
