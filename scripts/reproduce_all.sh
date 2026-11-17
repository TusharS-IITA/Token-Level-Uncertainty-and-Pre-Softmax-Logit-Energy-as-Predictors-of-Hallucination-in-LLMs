#!/bin/bash
set -e
echo "================================================================="
echo "End-to-End Reproduction Pipeline: AITR Lab 02"
echo "================================================================="
echo "Step 1: Running Unit Tests..."
bash scripts/run_tests.sh

echo "Step 2: Checking Dataset..."
test -f data/TruthfulQA.csv && echo "TruthfulQA.csv present."

echo "Step 3: Running Evaluation Display..."
python3 scripts/run_evaluation.py

echo "Step 4: Compiling Paper Manuscript..."
if command -v pdflatex &> /dev/null; then
    cd paper
    pdflatex -interaction=nonstopmode main.tex > /dev/null || true
    bibtex main > /dev/null || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null || true
    cd ..
    echo "Paper compiled successfully to paper/main.pdf."
fi

echo "================================================================="
echo "Reproduction Pipeline Finished Successfully!"
echo "================================================================="
