"""Unit tests for BLEU and ROUGE evaluation metrics."""

from src.captioncraft.metrics import compute_bleu, compute_rouge_l, evaluate_corpus

def test_compute_bleu_exact_match():
    cand = "a dog playing with a red ball in the park"
    refs = ["a dog playing with a red ball in the park"]
    scores = compute_bleu(cand, refs)
    assert scores["bleu_1"] == 1.0
    assert scores["bleu_4"] == 1.0

def test_compute_bleu_partial_match():
    cand = "a brown dog running in the park"
    refs = ["a black dog running in the garden", "a dog playing in the park"]
    scores = compute_bleu(cand, refs)
    assert scores["bleu_1"] > 0.5
    assert scores["bleu_4"] >= 0.0

def test_compute_bleu_brevity_penalty():
    cand = "dog"
    refs = ["a fast brown dog running in the park"]
    scores = compute_bleu(cand, refs)
    assert scores["bleu_1"] < 1.0

def test_compute_rouge_l_exact_match():
    cand = "a cat sits on the bench"
    refs = ["a cat sits on the bench"]
    rouge = compute_rouge_l(cand, refs)
    assert rouge["rougeL_fmeasure"] == 1.0

def test_evaluate_corpus():
    preds = ["a dog running", "a cat resting"]
    refs = [["a dog running in grass"], ["a cat resting on bed"]]
    res = evaluate_corpus(preds, refs)
    assert res["bleu_1"] > 0.0
    assert res["rougeL_fmeasure"] > 0.0
