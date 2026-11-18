#!/usr/bin/env python3
"""FinanceHarness headless one-shot research entry point (source checkout).

A thin wrapper so ``python main.py ...`` runs exactly the same CLI as the
installed ``financeharness`` / ``fh`` commands (``financeharness.cli:main``).
The full interface lives in that module:

    python main.py -p "your question" [--mode auto|research|analytical]
                   [--profile NAME] [--reader NAME] [--save out.json] [--quiet]
    echo "your question" | python main.py -p          # question via stdin
    python main.py --list                             # profiles + skills

The report is printed to stdout; progress goes to stderr (so stdout stays
pipeable). Exit code is 0 when the agent produced an answer, 1 otherwise.
"""

from financeharness.cli import main

if __name__ == "__main__":
    main()
