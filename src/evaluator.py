import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class Evaluator:
    @staticmethod
    def calculate_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(labels[in_bin] == 0)
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
        return float(ece)

    @staticmethod
    def calculate_metrics_with_ci(scores: np.ndarray, labels: np.ndarray, n_bootstraps: int = 1000, seed: int = 42):
        rng = np.random.RandomState(seed)
        auroc = roc_auc_score(labels, scores)
        prec, rec, _ = precision_recall_curve(labels, scores)
        auprc = auc(rec, prec)

        norm_conf = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        norm_conf = 1.0 - norm_conf
        ece = Evaluator.calculate_ece(norm_conf, labels)
        brier = np.mean(( (1.0 - norm_conf) - labels )**2)

        boot_aurocs = []
        boot_auprcs = []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(labels), len(labels))
            if len(np.unique(labels[idx])) < 2:
                continue
            boot_aurocs.append(roc_auc_score(labels[idx], scores[idx]))
            p, r, _ = precision_recall_curve(labels[idx], scores[idx])
            boot_auprcs.append(auc(r, p))

        auroc_ci = (float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5)))
        auprc_ci = (float(np.percentile(boot_auprcs, 2.5)), float(np.percentile(boot_auprcs, 97.5)))

        return {
            'auroc': float(auroc),
            'auroc_ci': auroc_ci,
            'auprc': float(auprc),
            'auprc_ci': auprc_ci,
            'ece': float(ece),
            'brier': float(brier)
        }
