#!/bin/bash
set -e
echo "Running automated unit testing suite..."
python3 -m unittest discover -s tests -p "test_*.py" -v
echo "All unit tests passed successfully."
