import torch
import torch.nn.functional as F

class SinglePassUncertaintyEngine:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def compute_token_entropies(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropies, min=0.0)

    @staticmethod
    def compute_logit_energy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_logits = logits[torch.arange(len(targets)), targets]
        return -target_logits

    def compute_sequence_scores(self, logits: torch.Tensor, targets: torch.Tensor, content_mask: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[torch.arange(len(targets)), targets]
        score_naive_logprob = -torch.mean(target_log_probs).item()

        entropies = self.compute_token_entropies(logits)
        if torch.sum(content_mask) > 0:
            score_content_entropy = (torch.sum(entropies * content_mask) / torch.sum(content_mask)).item()
            score_peak_entropy = torch.max(entropies * content_mask).item()
        else:
            score_content_entropy = torch.mean(entropies).item()
            score_peak_entropy = torch.max(entropies).item()

        energies = self.compute_logit_energy(logits, targets)
        score_logit_energy = torch.mean(energies).item()

        score_unified = (self.alpha * score_content_entropy) + (self.beta * score_peak_entropy) + (self.gamma * score_logit_energy)

        return {
            'naive_logprob': score_naive_logprob,
            'content_entropy': score_content_entropy,
            'peak_entropy': score_peak_entropy,
            'logit_energy': score_logit_energy,
            'unified_score': score_unified
        }
