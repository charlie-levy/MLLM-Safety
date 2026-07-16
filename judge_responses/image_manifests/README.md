# Image ↔ Response provenance — MM-SafetyBench-Tiny, VLS-Bench, HoliSafe

These manifests let you map **every model response back to the exact image and prompt it was generated from.**

## The files
| File | Rows | Dataset |
|---|---|---|
| `mmsafety_tiny_image_manifest.csv` | 168 | MM-SafetyBench-Tiny |
| `vls_bench_image_manifest.csv` | 500 | VLS-Bench |
| `holisafe_image_manifest.csv` | 494 | HoliSafe |

Columns: `dataset, idx, image_filename, image_path_newton, category, prompt`

## How to use it (the join key)
Every response record carries `idx`. **Join `(dataset, idx)` → this manifest** to get that sample's image, prompt, and category.

- **Corrupted responses** (Part 3): `results/part3_results/<dataset>_<corruption>_tis_responses.jsonl` — also carry `image_path` directly.
- **Clean responses** (Part 2): `results/exp_results/<dataset>_<model>_clean.jsonl` (`model` = `base` or `tis`).

`idx` is **identical across every clean and corrupted file** for a dataset (they all derive from the same materialized manifest), so the same `idx` is always the same image+prompt.

## The images
The PNGs are the **original source images** (re-encoded, renamed by `idx`), stored on Newton at the `image_path_newton` column:
```
/home/ch169788/experiments/part2/data/<dataset>/images/<idx:05d>.png
```
Pull any with `scp` (they are not in this repo).

## Exactly which subset (reproducible)
| Dataset | Source (HuggingFace) | Revision | Split / filter | Selection | N |
|---|---|---|---|---|---|
| MM-SafetyBench-Tiny | `PKU-Alignment/MM-SafetyBench` | `refs/convert/parquet` | **SD_TYPO** images; rows whose `id` ∈ **TinyVersion_ID_List** (13 scenarios) | official Tiny list | 168 |
| VLS-Bench | `Foreshhh/vlsbench` | `refs/convert/parquet` | config **default / train** | **balanced** across 6 categories, **seed 42** | 500 |
| HoliSafe | `etri-vilab/holisafe-bench` (gated) | `refs/convert/parquet` | rows where `type == "USU"` (unsafe-image / safe-text) | **balanced** across 7 categories, **seed 42** | 494 |

Tiny ID list: https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/TinyVersion_ID_List.json

## Note on `idx` vs. upstream IDs
`idx` (0…N-1) is our **internal sequential index** assigned at materialization — **not** the source dataset's native ID. The selection is deterministic (fixed revision + `seed 42`), so the exact `our_idx → upstream id` mapping is reproducible but was not stored here. If you need that explicit table (e.g., MM-SafetyBench's original `id`, or VLS/HoliSafe's source parquet row), it can be regenerated on request.
