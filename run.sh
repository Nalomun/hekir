#!/usr/bin/env bash
# Reproduce the analysis from data/companies.csv.
#
#   ./run.sh              embed -> analyze -> before/after -> co-movement -> figures
#   ./run.sh --fetch      also rebuild data/companies.csv from Wikipedia + yfinance first
#                         (the S&P 500 list and summaries will have changed since June 2026)
#   ./run.sh --qdrant     also load into Qdrant and verify neighbors
#                         (needs: docker run -p 6333:6333 qdrant/qdrant, and requirements-qdrant.txt)
#
# The co-movement step needs data/prices/close.csv.gz; it is downloaded
# (05_fetch_prices.py) if missing. Price data is gitignored and never committed.
set -euo pipefail
cd "$(dirname "$0")"

FETCH=0; QDRANT=0
for arg in "$@"; do
  case "$arg" in
    --fetch)  FETCH=1 ;;
    --qdrant) QDRANT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$FETCH" = 1 ]; then python src/01_fetch_data.py; fi
python src/02_embed.py
if [ "$QDRANT" = 1 ]; then python src/02b_load_qdrant.py; fi
python src/03_analyze.py
python src/compare_variants.py
[ -f data/prices/close.csv.gz ] || python src/05_fetch_prices.py
python src/06_comovement.py
python src/04_visualize.py
python src/make_fig4.py
