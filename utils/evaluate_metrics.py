import os
import sys
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from utils.benchmark_models import benchmark_all_models

def main():
    parser = argparse.ArgumentParser(description="CardioSignal-XAI Metrics & Benchmark Evaluator")
    parser.add_argument("--all", action="store_true", default=True, help="Benchmark all discovered models")
    args = parser.parse_args()

    print("[*] Launching CardioSignal-XAI Comprehensive Benchmark...")
    benchmark_all_models()

if __name__ == "__main__":
    main()