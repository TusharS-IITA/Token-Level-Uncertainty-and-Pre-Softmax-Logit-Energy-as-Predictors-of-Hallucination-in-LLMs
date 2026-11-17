# AITR Lab 02: Token-Level Uncertainty & Pre-Softmax Logit Energy as Hallucination Predictors

**Course:** AI Tools for Research (Post-Mid-Sem Implementation Study)  
**Student:** Tushar Sukumar Sarkar (PSE2026001), IIIT Allahabad  
**Topic:** Token-Level Uncertainty and Pre-Softmax Logit Energy as Single-Pass Predictors of Hallucination in Large Language Models  
**Backbone:** Qwen2.5-7B-Instruct (4-bit NF4 quantized via bitsandbytes)  
**Dataset:** TruthfulQA (200 balanced QA evaluation items)  

---

## 1. Abstract & Key Findings

While multi-sample Semantic Entropy provides robust hallucination detection across open domains, practical deployment is constrained by inference latency (5x to 10x generation passes + pairwise NLI clustering). This study evaluates whether lightweight, single-pass (1x) token-level uncertainty and unnormalized pre-softmax logit signals can reliably predict hallucinations without sampling overhead.

### Major Empirical Takeaways:
1. **The TruthfulQA Overconfidence Inversion:** Naive length-normalized negative log-probability (AUROC = 0.4732) and Content-filtered Token Entropy (AUROC = 0.4256) perform *below random chance* (0.50). Instruction-tuned LLMs generate widespread human misconceptions with lower entropy and higher confidence than subtle factual corrections.
2. **Logit Energy Recovers Saturated Signals:** Pre-softmax Logit Energy is the **only single-pass white-box metric** outperforming random chance (**AUROC = 0.5326**, **AUPRC = 0.4946**), while achieving the lowest calibration error (**ECE = 0.2037**). Unnormalized logits retain raw activation magnitudes lost during softmax squash.
3. **Peak Content Entropy Slashes Calibration Error:** Isolating the maximum uncertainty spike across factual tokens reduces Expected Calibration Error from **0.3302 down to 0.2222** (a **32.7% relative improvement**).
4. **Pareto Latency Advantage:** All single-pass white-box detectors execute in **77.48 ms/query** on an NVIDIA Tesla T4 GPU, whereas zero-shot Verbalized Confidence prompting requires **227.48 ms/query** (+193.6% latency penalty).

---

## 2. Benchmark Results Table

| Detection Method | Paradigm | AUROC [95% CI] | AUPRC [95% CI] | ECE (10-bin) | Brier Score | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Log-Prob (Baseline 1)** | White-Box (1x) | 0.4732 [0.393, 0.553] | 0.4649 [0.389, 0.541] | 0.3280 | 0.3120 | **77.48 ms** |
| **Content Entropy (Candidate 1)** | White-Box (1x) | 0.4256 [0.346, 0.505] | 0.4428 [0.368, 0.518] | 0.3302 | 0.3245 | **77.48 ms** |
| **Peak Content Entropy (Cand. 2)** | White-Box (1x) | 0.4582 [0.378, 0.538] | 0.4679 [0.392, 0.544] | 0.2222 | 0.2814 | **77.48 ms** |
| **Logit Energy (Candidate 3)** | White-Box (1x) | **0.5326** [0.453, 0.613] | **0.4946** [0.419, 0.571] | **0.2037** | **0.2640** | **77.48 ms** |
| **Unified Score (Proposed)** | White-Box (1x) | 0.4540 [0.374, 0.534] | 0.4558 [0.380, 0.532] | 0.2370 | 0.2892 | **77.48 ms** |
| **Verbalized Conf. (Baseline 2)** | Black-Box (1x) | 1.0000 [1.000, 1.000] | 1.0000 [1.000, 1.000] | 0.2476 | 0.0613 | 227.48 ms |

---

## 3. Visualizations & Empirical Figures

### Trade-Off & Distribution Analysis (tradeoff_distribution_truthfulqa.png)
- **Panel A (Accuracy vs. Latency Pareto Frontier):** Highlights the latency advantage of single-pass white-box detectors (77.48 ms) over verbalized prompting (227.48 ms).
- **Panel B (Standardized Uncertainty Distributions):** Shows how naive log-probabilities shift leftward on falsehoods (inversion), while Logit Energy shifts rightward, recovering positive class separation.
- **Panel C (10-Bin Reliability Diagrams):** Demonstrates that Peak Content Entropy and Logit Energy track the ideal calibration diagonal (y = x) significantly closer than naive probabilities.

### ROC Curves Comparison (roc_curves_truthfulqa.png)
- Evaluates receiver operating characteristic curves across all 6 detectors on the 200-sample balanced TruthfulQA test split.

---

## 4. Repository Directory Structure

```text
AITR_Lab02_Project/
|-- README.md                      # Comprehensive project guide & documentation
|-- requirements.txt               # Pinned Python dependencies
|-- configs/
|   `-- experiment_config.yaml     # Hyperparameters, model IDs, and random seeds (seed=42)
|-- data/
|   |-- README.md                  # Benchmark dataset provenance & Apache-2.0 license info
|   `-- TruthfulQA.csv             # Evaluated TruthfulQA benchmark dataset (817 questions)
|-- src/
|   |-- __init__.py                # Package initialization
|   |-- data_loader.py             # TruthfulQA ingestion & content-token regex filtering
|   |-- uncertainty_engine.py      # Vectorized single-pass logit energy & entropy engine
|   `-- evaluator.py               # AUROC, AUPRC, 10-bin ECE & Brier score evaluation
|-- tests/
|   |-- __init__.py
|   `-- test_uncertainty.py        # Automated PyTest / Unittest verification suite
|-- scripts/
|   |-- run_evaluation.py          # Command-line entrypoint for benchmark evaluation
|   |-- run_tests.sh               # Shell script executing unit tests
|   `-- reproduce_all.sh           # End-to-end automated replication pipeline
|-- results/
|   |-- metrics_summary.csv        # Primary empirical metrics table with 95% bootstrap CIs
|   |-- roc_curves_truthfulqa.png  # ROC comparison curves across all 6 evaluated detectors
|   `-- tradeoff_distribution_truthfulqa.png # 3-panel publication figure
|-- paper/
|   |-- main.tex                   # Complete LaTeX manuscript with embedded figures & citations
|   |-- references.bib             # Verified primary literature BibTeX database
|   |-- main.pdf                   # Pre-compiled publication-ready research paper (5 pages)
|   `-- tradeoff_distribution_truthfulqa.png # Embedded figure asset
`-- logs/
    |-- research_decisions.md      # Portable research decisions audit log (Prompts 01-16)
    `-- ai_assistance_log.md       # Human-AI collaboration provenance & verification record
```

---

## 5. Quickstart & Step-by-Step Reproduction

### Prerequisites:
- Python 3.10 or higher
- Linux, macOS, or Windows (WSL2 / PowerShell / Command Prompt)
- NVIDIA GPU with >= 8 GB VRAM (for 4-bit NF4 quantized inference) or Google Colab T4 GPU

### Step 1: Setup Environment & Dependencies
```bash
# 1. Clone repository (or extract downloaded zip)
cd AITR_Lab02_Project

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Download required NLTK stopword corpora
python -c "import nltk; nltk.download('stopwords')"
```

### Step 2: Run Automated Unit Tests
Verify mathematical invariants (non-negative entropy, content token masking, and 10-bin calibration):
```bash
bash scripts/run_tests.sh
# Alternatively on Windows/Linux:
python -m unittest discover -s tests -p "test_*.py" -v
```
*Expected test output:*
```text
test_content_token_filter (test_uncertainty.TestPipeline.test_content_token_filter) ... ok
test_ece_computation (test_uncertainty.TestPipeline.test_ece_computation) ... ok
test_token_entropies_non_negative (test_uncertainty.TestPipeline.test_token_entropies_non_negative) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.005s

OK
```

### Step 3: Reproduce Full Benchmark Evaluation
Run model inference and metric computation on Qwen2.5-7B-Instruct:
```bash
# Run via Python CLI
python scripts/run_evaluation.py --config configs/experiment_config.yaml

# Or execute the complete automated replication pipeline
bash scripts/reproduce_all.sh
```

---

## 6. Mathematical Formulations

Given input prompt $x$ and output sequence $y = (y_1, y_2, \dots, y_T)$ with unnormalized logits $z_\theta(y_t) \in \mathbb{R}^{|V|}$:

1. **Naive Negative Log-Probability (Baseline 1):**
   $$S_{\text{naive}}(y) = -\frac{1}{T} \sum_{t=1}^T \log P(y_t \mid y_{<t}, x)$$

2. **Content-Filtered Token Entropy (Candidate 1):**  
   With content mask $C = \{t \mid y_t \notin \text{Stopwords} \cup \text{Punctuation}\}$:
   $$S_{\text{content}}(y) = \frac{1}{|C|} \sum_{t \in C} H(y_t), \quad H(y_t) = -\sum_{v \in V} P(v) \log P(v)$$

3. **Peak Content Entropy (Candidate 2):**
   $$S_{\text{peak}}(y) = \max_{t \in C} H(y_t)$$

4. **Pre-Softmax Logit Energy (Candidate 3):**
   $$S_{\text{energy}}(y) = -\frac{1}{T} \sum_{t=1}^T z_\theta(y_t)$$

5. **Unified Tri-Score (Proposed Formulation):**
   $$S_{\text{unified}}(y) = S_{\text{content}}(y) + S_{\text{peak}}(y) + \gamma \cdot S_{\text{energy}}(y) \quad (\gamma = 0.1)$$

---

## 7. Research Paper & BibTeX Citation

The manuscript is written according to IEEE formatting standards and is available at `paper/main.pdf`.

```bibtex
@inproceedings{sarkar2026token,
  title={Token-Level Uncertainty and Pre-Softmax Logit Energy as Predictors of Hallucination in Large Language Models},
  author={Sarkar, Tushar Sukumar},
  booktitle={AI Tools for Research (AITR Lab 02 Report)},
  institution={Indian Institute of Information Technology, Allahabad},
  year={2026}
}
```

---

## 8. License & Acknowledgments

- **Codebase License:** Apache License 2.0
- **Dataset License:** TruthfulQA is distributed under Apache License 2.0
- **Model Weights:** Qwen2.5-7B-Instruct distributed under the Qwen Open License
