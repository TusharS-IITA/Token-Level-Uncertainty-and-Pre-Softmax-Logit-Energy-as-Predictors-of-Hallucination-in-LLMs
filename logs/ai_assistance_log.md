# Human-AI Collaboration & Assistance Audit Log (Prompt 18)

**Course:** AI Tools for Research (Post-Mid-Sem Implementation Study)  
**Student:** Tushar Sukumar Sarkar (PSE2026001), IIIT Allahabad  
**Project:** Token-Level Uncertainty and Pre-Softmax Logit Energy as Single-Pass Predictors of Hallucination in Large Language Models  

---

## 1. Governance & Compliance Declaration
All research activities, code generation, and manuscript drafts were conducted under strict human-in-the-loop oversight. AI tools were utilized exclusively as interactive research assistants for literature synthesis, prototyping, LaTeX authoring, and documentation generation. All numerical findings, formulas, and conclusions reflect verified executions on the designated NVIDIA Tesla T4 GPU benchmark.

---

## 2. Interaction Provenance Matrix

| Phase & Prompt | AI Assistant Contribution | Human Author Verification & Corrective Action |
| :--- | :--- | :--- |
| **Phase I (Prompts 01-05)** | Initial literature review structuring, 5 testable gap matrix draft, resource budgeting. | Filtered research scope from multi-sample NLI to single-pass efficiency; confirmed Qwen2.5-7B NF4 feasibility on Colab. |
| **Phase II (Prompts 06-07)** | Generated Colab GPU sprint script (prompt_06_colab_sprint.py); drafted unit tests. | Fixed FileNotFoundError for TruthfulQA fallback; verified seed determinism (seed=42) and GPU allocation. |
| **Phase III (Prompts 08-11)** | Modularized code into src package; produced baseline calibration analysis. | Audited unit test suite (4/4 tests passed); fixed escaped LaTeX characters in debug scripts. |
| **Phase IV (Prompts 12-14)** | Computed 95% bootstrap CIs; generated 3-panel figure script (tradeoff_distribution_truthfulqa.png). | Validated TruthfulQA overconfidence inversion; verified math formulation of unnormalized logit energy. |
| **Phase V (Prompts 15-17)** | Authored IEEE two-column LaTeX manuscript (main.tex) and repository documentation (README.md). | Resolved LaTeX escaping errors; embedded 3-panel evaluation figure with literature citations. |

---

## 3. Academic Integrity Confirmation
I confirm that:
1. No ungrounded or fabricated empirical data was included.
2. All citations reflect genuine published literature indexed in ACL Anthology, IEEE, Nature, and arXiv.
3. The codebase has been verified and tested for standalone reproducibility.
