# Data provenance

Both files are copied/distilled from the public trajectory-judge release
(Zenodo DOI 10.5281/zenodo.21797926), 2026-08-05.

- `verdicts.jsonl` - verbatim raw verdicts: 2,000 rows, five judges x 400
  trajectories. Fields used here: `trajectory_id`, `judge_id`, `faulty`.
  md5 `31c04822dc3258b081b254e446181f88`.
- `gold.jsonl` - one row per trajectory with the construction-time label
  (`faulty`, `failure_type`, `outcome_correct`).
  md5 `aaf136b0b26c494de549f4f82dcbd759`.

The gold labels exist because faults were injected by construction in the
source study; the Dawid-Skene fits never see them - they score the
estimates afterwards.
