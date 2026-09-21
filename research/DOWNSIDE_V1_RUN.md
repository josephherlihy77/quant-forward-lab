# Running DOWNSIDE_V1 Research Pass 001

1. In the Kaggle notebook/session containing `/kaggle/working/v4_clean_research_dataset.parquet`, add or upload `research/downside_v1_pass001.py`.
2. Run: `python downside_v1_pass001.py`
3. Preserve the entire `/kaggle/working/downside_v1/` output directory.
4. Do not change BALANCED_FORWARD_V1 or Snapshot #001.

Outputs:
- `baseline_results.csv` — validation/test precision, recall and bearish-basket forward excess returns.
- `research_frame.parquet` — auditable features/labels/splits.
- `manifest.json` — configuration and output hash.

Pass 001 deliberately uses a fixed equal-weight composite rather than tuning weights. Its market-relative target uses the monthly cross-sectional median as a temporary market proxy. Before any model is promoted or frozen, replace that proxy with an explicit adjusted SPY total-return series and rerun the untouched evaluation.
