"""``python -m backtest``: see ``backtest.cli``."""

import os

# Fixed before numpy and scikit-learn start their thread pools (design §8). Results are
# identical at 1, 2 and 8 threads (measured); the count is recorded with every experiment.
os.environ.setdefault("OMP_NUM_THREADS", "8")

from backtest.cli import main

raise SystemExit(main())
