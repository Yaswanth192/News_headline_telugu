## Slide 1: Title

- Telugu News Headline Generation
- Final Project Report
- Generated on 2026-10-13

---

## Slide 2: Problem & Motivation

- Automatic headline generation for Telugu news
- Support low‑resource settings
- Improve editorial efficiency

---

## Slide 3: Objectives

- Train and compare mT5‑small & IndicBART
- Build relevance classifier
- Low‑resource experiments
- Report findings

---

## Slide 4: Dataset

- Source: saidines12/telugu_news_dataset
- ~83k train, 5.9k val, 12.5k test
- Duplicate & leakage analysis included

---

## Slide 5: Model Architecture

- Seq2Seq mT5‑small (300M params)
- IndicBART (244M params)
- BERT relevance classifier

---

## Slide 6: Training Setup

- bfloat16, batch=1, grad_accum=16
- 50 steps (full), 75 steps (low‑resource)
- Runs on RTX 3050 (4 GB VRAM)

---

## Slide 7: Results – Model Comparison

- mT5‑small ROUGE‑1=0.1025, latency=0.27 s
- IndicBART ROUGE‑1=0.0109, latency=2.62 s

---

## Slide 8: Low‑Resource Findings

- 5 % data retains ~85 % ROUGE‑1 of full data
- Training time comparable across tiers

---

## Slide 9: Relevance Classification

- mBERT Macro‑F1=0.5114 (Accuracy=0.5238)
- TF‑IDF+LR Macro‑F1=0.4775

---

## Slide 10: Gemma 2 Feasibility

- Blocked: HF gated repo, missing bitsandbytes/peft, VRAM insufficient

---

## Slide 11: Streamlit Demo

- Headline generation page
- Relevance analysis page
- Dashboard with metrics

---

## Slide 12: Limitations

- Very short training (50‑75 steps)
- Small evaluation sample sizes
- No human evaluation
- Gemma 2 not trained

---

## Slide 13: Conclusion

- mT5‑small is the pragmatic choice for low‑resource Telugu headline generation
- Future work: longer training, human studies, larger LLMs

---

