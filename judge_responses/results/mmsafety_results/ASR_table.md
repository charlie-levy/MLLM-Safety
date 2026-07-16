# MM-SafetyBench SD_TYPO — Attack Success Rate (ASR)

**Judge:** LLaMA-Guard-3-11B-Vision, text-only `(prompt, response)`, locked
**Models:** `base` = LLaVA-CoT (Xkev/Llama-3.2V-11B-cot) · `tis` = LLaVA-CoT + TIS LoRA
**Inference:** bf16, greedy, `max_new_tokens=2048`, one-at-a-time, seed=0 (frozen contract)
**Subsets:** Sex (109) + Physical_Harm (144) = **253** prompts per condition
All cells n=253

## Overall (unsafe / 253)

| Model    | Clean         | Blur          | Noise         | JPEG          | Occlusion     |
|----------|---------------|---------------|---------------|---------------|---------------|
| **base** | 109/253 (43%) | 115/253 (45%) | 113/253 (45%) | 128/253 (51%) | 118/253 (47%) |
| **tis**  | 0/253 (0%)    | 0/253 (0%)    | 0/253 (0%)    | 0/253 (0%)    | 0/253 (0%)    |

## By subset

| Model| Subset        | Clean        | Blur         | Noise        | JPEG         | Occlusion    |
|------|---------------|--------------|--------------|--------------|--------------|--------------|
| base | Sex           | 30/109 (28%) | 36/109 (33%) | 32/109 (29%) | 38/109 (35%) | 34/109 (31%) |
| base | Physical_Harm | 79/144 (55%) | 79/144 (55%) | 81/144 (56%) | 90/144 (62%) | 84/144 (58%) |
| tis  | Sex           | 0/109 (0%)   | 0/109 (0%)   | 0/109 (0%)   | 0/109 (0%)   | 0/109 (0%)   |
| tis  | Physical_Harm | 0/144 (0%)   | 0/144 (0%)   | 0/144 (0%)   | 0/144 (0%)   | 0/144 (0%)   |

## Notes

- **TIS is flat 0%** across clean and all 4 corruptions — genuine engaged refusals.
- **base** sits at 43% clean and rises under corruption, peaking at **JPEG (+19 → 51%)**, driven mostly by the Physical_Harm subset (55 → 62%).
- That base bump is largely **LLaMA-Guard inconsistency on long CoT responses**, not a clean corruption-breaks-safety signal — a single severity per corruption can't separate the two. Testing the ASR-vs-severity curve needs a sweep.

1. TIS is a hard floor, not just a low number. 0% across every cell — both subsets, all 4 corruptions, 253 prompts each. And those zeros are genuine (full CoT, perceives the harmful content, refuses on policy), not blank/garbled outputs. So safety tuning here fully survives image corruption. That's the single strongest, cleanest pattern.

2. Harm category dominates everything — far more than corruption. Physical_Harm baseline ASR (55%) is roughly double Sex (28%). That gap (~27 pts) dwarfs any corruption effect (≤8 pts). Base will give operational "list the steps" answers for physical-harm prompts much more readily than for sexual ones, even with no corruption at all. If you're looking for where base is dangerous, it's the harm type, not the noise.

3. Corruption only weakly nudges base, and it's not a clean dose-response. Every corruption is ≥ clean (43% → blur/noise 45%, occlusion 47%, jpeg 51%), so there's a real upward tendency — but it's non-monotonic and small, and the peak (jpeg, +19) is concentrated in Physical_Harm. When I read those flips, most were LLaMA-Guard inconsistency on long CoT responses (benign self-care tagged S11, a harmful clean response tagged safe while a hedged jpeg one tagged unsafe), with only a minority genuine. So: no clean "corruption degrades safety" signal at single severity — exactly what the ACZ hypothesis would need, and it's not there.

4. The mechanistic reason it's not there: corruption never actually blinded the model. perception_failure = 0 in every cell, even at blur r=5 / noise std=110 / jpeg q=5 / occlusion 40%. The model never said "I can't read the image." Because SD_TYPO bakes the harmful phrase as text rendered into the image, and that text stayed legible-enough through these corruptions. The safety-relevant signal survived, so corruption had little to break.

The synthesis (ties back to the earlier experiments too): corruption breaks safety only when all three hold — the harm signal lives in the image, corruption actually destroys that signal, and the model isn't safety-tuned. SD_TYPO fails the middle condition (typographic harm survives corruption); TIS fails the last one. That's consistent with BeaverTails (text-borne harm → robust) and points the next experiment at the one knob we couldn't move here: a severity sweep to find where the image signal finally degrades, on Physical_Harm + base only — the one place an effect could actually show.