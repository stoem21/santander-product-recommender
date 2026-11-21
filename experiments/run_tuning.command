#!/bin/bash
# Double-click in Finder to run the hyperparameter search natively on this Mac (no time limit, all performance cores).
# Safe to run again: it resumes where it stopped.
cd "$(dirname "$0")/.." || exit 1          # repository root
echo "== Santander tuning in: $(pwd)"
if [ ! -d .venv ]; then
  python3 -m venv .venv || { echo "Could not create a Python environment (is python3 installed?)"; read -p "Press Enter to close"; exit 1; }
fi
source .venv/bin/activate
pip install --upgrade pip > /dev/null
pip install -r experiments/requirements_tune.txt || { echo "Package install failed"; read -p "Press Enter to close"; exit 1; }
python3 -c "import lightgbm, xgboost, torch" || {
  echo; echo "Import failed. If the message mentions libomp, run:  brew install libomp   then double-click this file again."
  read -p "Press Enter to close"; exit 1; }
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
echo "== using $CORES threads"
mkdir -p analysis_output/tuning
if [ ! -f analysis_output/tuning/.mac_started ]; then          # first run on the Mac: redo everything on one machine
  rm -f analysis_output/tuning/results.csv analysis_output/tuning/units_done.csv
  touch analysis_output/tuning/.mac_started
fi
echo "== started $(date)" | tee -a tune_log.txt
# One process per model: torch bundles its own libomp, lightgbm/xgboost load Homebrew's libomp; two copies in one
# process abort with "OMP: Error #15" (happened 2026-11-20 when the NN started right after XGBoost).
for M in lgbm xgb nn; do
  echo "== model $M" | tee -a tune_log.txt
  TUNE_THREADS=$CORES caffeinate -i python3 -m experiments.tune run --budget 1000000 --only $M 2>&1 | grep --line-buffered -v "Warning" | tee -a tune_log.txt
done
echo "== finished $(date)" | tee -a tune_log.txt
read -p "Done. Press Enter to close this window"
