"""Evaluation metrics for image captioning (BLEU-1 to BLEU-4 and ROUGE-L)."""

import math
from collections import Counter
from typing import List, Dict, Union

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-gram counts from token sequence."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_bleu(
    candidate: Union[str, List[str]],
    references: List[Union[str, List[str]]],
    max_n: int = 4
) -> Dict[str, float]:
    """Compute modified n-gram precision and BLEU-1 through BLEU-4 scores with brevity penalty.
    
    Args:
        candidate: Predicted caption string or list of tokens.
        references: List of reference caption strings or lists of tokens.
        max_n: Maximum n-gram length (default 4).
    """
    cand_tokens = candidate.split() if isinstance(candidate, str) else list(candidate)
    ref_token_lists = [r.split() if isinstance(r, str) else list(r) for r in references]

    if not cand_tokens:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    # Brevity Penalty
    cand_len = len(cand_tokens)
    ref_lens = [len(r) for r in ref_token_lists]
    # Closest reference length
    closest_ref_len = min(ref_lens, key=lambda rlen: (abs(rlen - cand_len), rlen))
    
    if cand_len > closest_ref_len:
        bp = 1.0
    elif cand_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / cand_len))

    precisions = []
    scores = {}

    for n in range(1, max_n + 1):
        cand_ngrams = _get_ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            scores[f"bleu_{n}"] = 0.0
            continue

        # Max counts in any reference
        max_ref_counts: Counter = Counter()
        for ref in ref_token_lists:
            ref_ngrams = _get_ngrams(ref, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        clipped_count = sum(min(count, max_ref_counts[ngram]) for ngram, count in cand_ngrams.items())
        total_count = sum(cand_ngrams.values())
        p_n = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(p_n)

        # Cumulative geometric mean
        if all(p > 0 for p in precisions):
            log_prec_sum = sum((1.0 / n) * math.log(p) for p in precisions)
            bleu_n = bp * math.exp(log_prec_sum)
        else:
            bleu_n = 0.0
        scores[f"bleu_{n}"] = round(bleu_n, 4)

    return scores

def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(
    candidate: Union[str, List[str]],
    references: List[Union[str, List[str]]],
    beta: float = 1.2
) -> Dict[str, float]:
    """Compute ROUGE-L (LCS-based recall, precision, and F1 measure) against multiple references."""
    cand_tokens = candidate.split() if isinstance(candidate, str) else list(candidate)
    ref_token_lists = [r.split() if isinstance(r, str) else list(r) for r in references]

    if not cand_tokens or not ref_token_lists:
        return {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_fmeasure": 0.0}

    best_f = 0.0
    best_p = 0.0
    best_r = 0.0

    m = len(cand_tokens)
    for ref in ref_token_lists:
        n = len(ref)
        lcs = compute_lcs_length(cand_tokens, ref)
        if lcs == 0:
            continue
        p = lcs / m if m > 0 else 0.0
        r = lcs / n if n > 0 else 0.0
        
        # F_beta measure
        if p + r > 0:
            f = ((1 + beta**2) * p * r) / (beta**2 * p + r)
        else:
            f = 0.0

        if f > best_f:
            best_f = f
            best_p = p
            best_r = r

    return {
        "rougeL_precision": round(best_p, 4),
        "rougeL_recall": round(best_r, 4),
        "rougeL_fmeasure": round(best_f, 4)
    }

def evaluate_corpus(
    predictions: List[str],
    references_list: List[List[str]]
) -> Dict[str, float]:
    """Calculate mean corpus-level BLEU-1, BLEU-4, and ROUGE-L across test samples."""
    if not predictions:
        return {"bleu_1": 0.0, "bleu_4": 0.0, "rougeL_fmeasure": 0.0}

    total_bleu_1 = 0.0
    total_bleu_4 = 0.0
    total_rouge_l = 0.0
    n = len(predictions)

    for pred, refs in zip(predictions, references_list):
        b = compute_bleu(pred, refs, max_n=4)
        r = compute_rouge_l(pred, refs)
        total_bleu_1 += b["bleu_1"]
        total_bleu_4 += b["bleu_4"]
        total_rouge_l += r["rougeL_fmeasure"]

    return {
        "bleu_1": round(total_bleu_1 / n, 4),
        "bleu_4": round(total_bleu_4 / n, 4),
        "rougeL_fmeasure": round(total_rouge_l / n, 4),
    }
