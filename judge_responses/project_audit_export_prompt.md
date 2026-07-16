# Full Project Audit & Export Prompt (No-SSH Version)

Your coding agent cannot SSH into Newton directly — it can only give you commands
to run yourself, then work from what you paste back or copy down. The prompt below
is written around that constraint: it makes the agent produce exact copy-paste
commands at every step that needs Newton, and builds one final zip you hand to
your advisors.

Run the "PROMPT" section in your coding agent from your local project directory
(wherever your GitHub clone / local work lives). It already has that filesystem —
Newton is the only thing it needs your hands for.

---

## PROMPT (copy from here down)

You are doing a full audit and export of my REU research project so I can hand my
advisors one complete package. This is read-only discovery + reorganization —
do not modify, retrain, or delete anything, anywhere, including on Newton.

You do NOT have SSH/direct access to Newton (the HPC cluster). For anything that
requires touching Newton, do not attempt to connect yourself — instead, generate
the exact shell commands for me to run in my own Newton SSH session, tell me what
to do with the output (paste it back to you, or run a follow-up scp command you
also give me), and wait for my reply before continuing. Never guess at Newton's
contents.

### Project context (so you don't have to re-derive it)

Title: *Evaluating the Robustness of MLLM Safety Alignment under Image Corruptions*
(targeting WACV-2027).

Core question: do everyday image corruptions (blur, noise, snow, fog, frost, JPEG
compression, pixelation) weaken safety alignment in multimodal LLMs, and can it be
restored (training or prompting)?

Models: LLaVA-CoT (11B), LLaVA-CoT + TIS (LoRA-tuned), Llama-3.2-11B-Vision-Instruct,
Qwen2.5-VL (7B), LlamaV-o1 (11B), R1-Onevision (7B, incl. no-think variant),
LLaVA-1.5-7B + VLGuard.

Safety/utility mechanisms: TIS (Think-in-Safety) LoRA training, MSRAlign, LoRA
ablations.

Datasets/benchmarks: ScienceQA (utility), FigStep, BeaverTails-V, SIUO, Holisafe,
MM-SafetyBench, XSTest, MMSA.

Metrics: ASR (attack success rate, via Llama-3/Llama-Guard-3/GPT-4o judges), ORR
(over-refusal rate), HR (harmfulness rate, reasoning vs. conclusion), utility
accuracy (ScienceQA/MathVista), training loss curves.

Infra: LLaMA-Factory for fine-tuning, checkpoints trained on Newton, local runs for
smaller evals. GitHub repo: https://github.com/charlie-levy/MLLM-Safety.git

Timeline: weekly presentations Week 1 (05/28/26) through Week 8 (07/14/26,
current), each summarizing that week's runs, plots, and tables.

### Phase 0 — What you can do yourself, right now

1. Clone/pull the GitHub repo directly (`git clone https://github.com/charlie-levy/MLLM-Safety.git`).
2. Inventory everything reachable from my local filesystem: the repo clone, any
   other local project folders (scratch, sandbox, duplicated copies, Downloads,
   Desktop — search broadly by filename pattern, not just the obvious folder),
   and any local wandb/tensorboard/runs/logs/outputs/results/checkpoints
   directories.
3. Read the 8 weekly presentation files I've given you as a source of *claims* to
   verify against real artifacts — not as the artifacts themselves.

Build a running manifest (path, file type, mtime, which experiment/week it likely
belongs to, one-line description) as you go — this becomes `00_manifest.csv` later.

### Phase 1 — Newton discovery (requires me)

Do not try to connect to Newton. Instead, give me a single block of read-only
discovery commands to run in my own Newton SSH session, e.g. (adjust to my actual
paths/username once I confirm them):

```bash
# run on Newton, then paste the full output back
echo "== home dir ==" && find ~ -maxdepth 4 \
  \( -iname "*checkpoint*" -o -iname "*.json" -o -iname "*.csv" -o -iname "*.jsonl" \
     -o -iname "*wandb*" -o -iname "*tensorboard*" -o -iname "*slurm*" \
     -o -iname "*result*" -o -iname "*log*" -o -iname "*.png" -o -iname "*.pdf" \) \
  -printf "%T@ %s %p\n" 2>/dev/null | sort -n

echo "== scratch/project space (adjust path) ==" && find /scratch/$USER -maxdepth 5 \
  \( -iname "*checkpoint*" -o -iname "*.json" -o -iname "*.csv" -o -iname "*.jsonl" \
     -o -iname "*wandb*" -o -iname "*tensorboard*" -o -iname "*slurm*" \
     -o -iname "*result*" -o -iname "*log*" \) \
  -printf "%T@ %s %p\n" 2>/dev/null | sort -n

echo "== disk usage by directory (sanity check for huge stuff) ==" && \
  du -h --max-depth=2 ~ /scratch/$USER 2>/dev/null | sort -rh | head -50
```

Ask me to paste the output back. From that listing, decide what's small enough
and relevant enough to pull down (configs, logs, JSON/CSV results, plots — NOT
multi-GB checkpoint weights) and give me exact `scp` commands to run locally, e.g.:

```bash
# run on MY machine (not Newton), one line per file/folder you want
mkdir -p ~/mllm_audit_pull/newton
scp newton:/path/to/results/foo.json ~/mllm_audit_pull/newton/
scp -r newton:/path/to/logs/week6_run ~/mllm_audit_pull/newton/
```

For anything too large to pull (checkpoint weights, huge raw logs), don't try to
copy it — just record its path, approximate size (from the `du` output), and a
one-line description in the manifest, and note in the final summary that the file
itself lives on Newton at that path and wasn't included in the zip.

Repeat Phase 1 as many times as needed (discovery → decide what to pull → give me
scp commands → I run them and confirm) until Newton is covered. Don't assume one
round is enough — check subdirectories the first pass didn't reach.

### Phase 2 — What counts as a "result" worth pulling in

- Raw eval outputs: JSON/CSV/JSONL logs of ASR, ORR, HR, accuracy per
  model × corruption × severity × dataset
- Aggregated result tables (anything resembling the Clean/Blur20/Blur40/JPEG/
  Pixelate tables from the decks)
- Plots/figures (severity curves, bar charts, qualitative examples, attention
  maps, reasoning-trace screenshots)
- Training artifacts: loss curves, checkpoint *metadata* (step count, trained-
  param count, final loss), LoRA configs — not the weight files themselves
- Qualitative examples/transcripts (clean-vs-corrupted refuse→comply cases)
- Any draft writeup, notes, or paper draft (WACV submission material)
- Config files defining each experiment (corruption type/severity, model,
  dataset, judge model used)

### Phase 3 — Cross-reference against the decks

For each week's presentation, list the specific claims/numbers/plots shown, and
try to match each to a real file you've inventoried or pulled. Flag anything shown
in a deck that you cannot source — that's exactly what advisors will ask about, so
surface it rather than paper over it.

### Phase 4 — Build ONE zip file for my advisors

Structure everything (local + whatever got pulled from Newton) into a staging
folder, then zip it as a single file:

```
mllm_safety_audit_export/
  00_START_HERE.md              <- see below
  00_manifest.csv
  week_01_intro/
  week_02_project_setup/
  week_03_pipeline_setup/
  week_04_severity_orr/
  week_05_beavertails_siuo/
  week_06_where_safety_breaks/
  week_07_finetuning_tis/
  week_08_full_model_sweep/
  raw_logs/                     <- anything not cleanly mapped to one week
  figures/
  repo_snapshot/                <- relevant parts of the cloned GitHub repo
  newton_pulled/                <- what got scp'd down in Phase 1
```

```bash
cd ~ && zip -r mllm_safety_audit_export.zip mllm_safety_audit_export/
```

Keep the zip advisor-sized: no checkpoint weights, no multi-GB raw dumps — those
stay referenced-by-path-only in the manifest with a note that they live on Newton.
Target is "everything an advisor would want to look at," not "every byte that
exists."

`00_START_HERE.md` should be a plain, direct summary (no fluff) covering:
- What experiments exist and what each tested, mapped to the week-by-week
  narrative above
- Final headline results table(s) — the most complete/current version of the
  model × corruption × metric grid
- What's confirmed solid (e.g., Week 7's loss-curve match to the mentor's
  reproduction) vs. still in progress
- Explicit list of gaps: deck claims with no located source, or "next steps"
  that never got artifacts
- A "where things live" index: what's in the zip vs. what's noted-but-not-
  included (with Newton paths) vs. what genuinely couldn't be found

### Constraints

- Read-only on all source files/checkpoints/configs everywhere, including Newton.
  Never modify, retrain, or re-run anything, and never delete anything from its
  original location.
- Never attempt an SSH/remote connection yourself — always hand me the command.
- If something needs credentials you don't have, stop and tell me — don't guess
  or silently skip it.
- When the zip is done, tell me its path and size first, before anything else.

---

## Before you run this

- Have your Newton username/hostname and typical project path ready — the agent
  will ask, or you can front-load it (e.g., "my Newton alias is `newton`, my
  project space is `/scratch/clevy531/mllm-safety`").
- Expect a few back-and-forth rounds in Phase 1 — that's the point of the no-SSH
  workflow, not a failure of it.
- If the zip is still huge because of unavoidable local files (e.g. big local
  eval logs), that's fine — "huge zip" was the brief. Just make sure it's the
  right huge, not padded with checkpoint weights nobody will open.
