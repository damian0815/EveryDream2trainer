#!/bin/bash
# Run all unit and integration tests in the everydream2trainer conda environment

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate everydream2trainer

# Install test dependencies if needed
pip install -r requirements-test.txt || true
pip install pytest

# Run all tests (unit and integration)
pytest test/ --maxfail=3 --disable-warnings -v
pytest test_minsnrgamma.py --maxfail=3 --disable-warnings -v
if [ -f scripts/test_tokenizer.py ]; then
  pytest scripts/test_tokenizer.py --maxfail=3 --disable-warnings -v
fi

echo "All tests completed."

