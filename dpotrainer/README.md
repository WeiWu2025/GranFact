# Objective Margin DPO Trainer

This directory contains the custom multi-objective margin-DPO trainer used by
the Grain-HAL DPO experiments.

Important: this is a **ms-swift patch file**, not a standalone trainer package.
`objective_margin_dpo_trainer.py` expects to live inside ms-swift's
`swift/rlhf_trainers/` package because it imports the local ms-swift trainer via:

```python
from .dpo_trainer import DPOTrainer
```

## What it does

`ObjectiveMarginDPOTrainer` extends the local ms-swift `DPOTrainer` by:

- reading pair metadata such as `objective_id`, `score_diff`, and `margin`;
- removing those metadata fields before model forward;
- reusing `DPOTrainer.concatenated_forward` for policy/reference log-probs;
- applying sigmoid DPO with an optional runtime margin;
- reducing losses with objective-aware weighting;
- optionally using an objective interleaving sampler that works with
  `per_device_train_batch_size=1`.

In the paper, this custom trainer is referred to as `RPDPOTrainer`. In this
release, the corresponding ms-swift patch class is named
`ObjectiveMarginDPOTrainer` and is registered through the `objective_margin_dpo`
RLHF type in the examples below.

Expected dataset fields:

```json
{
  "objective_id": 1,
  "objective_type": "truth",
  "score_diff": 0.12
}
```

Objective IDs:

| ID | Objective |
|---:|-----------|
| 1 | truth |
| 2 | quality_f1 |

`stage_type` is also accepted as a backward-compatible alias for
`objective_id`.

## Integration steps

### 1. Locate your active ms-swift installation

```bash
python - <<'PY'
import pathlib, swift
print(pathlib.Path(swift.__file__).resolve())
PY
```

The target directory is usually:

```text
/path/to/site-packages/swift/rlhf_trainers/
```

### 2. Copy the trainer

```bash
cp dpotrainer/objective_margin_dpo_trainer.py \
  /path/to/site-packages/swift/rlhf_trainers/objective_margin_dpo_trainer.py
```

### 3. Export it from `swift/rlhf_trainers/__init__.py`

Add the equivalent of:

```python
from .objective_margin_dpo_trainer import ObjectiveMarginDPOTrainer
```

If your ms-swift version uses lazy import mappings, also add the module/class to
that mapping, for example:

```python
'objective_margin_dpo_trainer': ['ObjectiveMarginDPOTrainer']
```

### 4. Register the RLHF type

Find the ms-swift code that maps `rlhf_type` to trainer classes and register a
new type without replacing normal DPO. The exact file varies by ms-swift
version. Search with:

```bash
python - <<'PY'
import pathlib, swift
root = pathlib.Path(swift.__file__).resolve().parent
for p in root.rglob('*.py'):
    text = p.read_text(encoding='utf-8', errors='ignore')
    if 'DPOTrainer' in text and 'rlhf_type' in text:
        print(p)
PY
```

Then add the equivalent of:

```python
from swift.rlhf_trainers import ObjectiveMarginDPOTrainer

RLHF_TRAINER_MAP['objective_margin_dpo'] = ObjectiveMarginDPOTrainer
```

or adapt the corresponding selection logic in your ms-swift version.

### 5. Verify the patch

```bash
python - <<'PY'
import inspect, pathlib
from swift.rlhf_trainers import ObjectiveMarginDPOTrainer
print(ObjectiveMarginDPOTrainer)
print(pathlib.Path(inspect.getfile(ObjectiveMarginDPOTrainer)).resolve())
PY
```

During training, logs should contain:

```text
[ObjectiveMarginDPOTrainer] enabled:
[ObjectiveInterleavingSampler] Found truth=..., quality_f1=...
train/objective_truth_loss
train/objective_quality_loss
train/objective_margin_mean
```

If these markers are absent, the run is probably not using this trainer.

## Environment variables

The trainer uses environment variables so it can work before custom CLI
dataclasses are registered in ms-swift.

| Variable | Default | Meaning |
|---|---:|---|
| `MO_DPO_QUALITY_WEIGHT` | `0.5` | Quality objective coefficient in `sum` mode; fallback value for `MO_DPO_QUALITY_LAMBDA`. |
| `MO_DPO_QUALITY_LAMBDA` | same as `MO_DPO_QUALITY_WEIGHT` | Quality objective coefficient in `convex` mode. Must be in `[0, 1]`. |
| `MO_DPO_OBJECTIVE_WEIGHT_MODE` | `convex` | `convex` or `sum`. |
| `MO_DPO_USE_OBJECTIVE_WEIGHT` | `true` | Whether to reduce truth/quality losses separately. |
| `MO_DPO_USE_MARGIN` | `true` | Whether to apply runtime margin. |
| `MO_DPO_MARGIN_SOURCE` | `auto` | `auto`, `margin`, or `score_diff`. |
| `MO_DPO_MARGIN_BETA_MODE` | `inside` | `inside`: `-logsigmoid(beta * (logits - margin))`; `outside`: `-logsigmoid(beta * logits - margin)`. |
| `MO_DPO_MARGIN_MODE` | `linear` | `linear`, `sqrt`, `log`, `constant`, `none`, `off`, or `false`. |
| `MO_DPO_TRUTH_MARGIN_SCALE` | `2.0` | Margin scale for truth objective. |
| `MO_DPO_QUALITY_MARGIN_SCALE` | `1.0` | Margin scale for quality objective. |
| `MO_DPO_MARGIN_MIN` | `0.0` | Minimum clipped margin. |
| `MO_DPO_MARGIN_MAX` | `1.0` | Maximum clipped margin. |
| `MO_DPO_STRICT_FIELDS` | `true` | Fail on missing/unknown objective fields. |
| `MO_DPO_USE_OBJECTIVE_SAMPLER` | `true` | Use objective interleaving sampler. |
| `MO_DPO_TRUTH_PER_ACCUM` | `8` | Truth samples per virtual accumulation window. |
| `MO_DPO_QUALITY_PER_ACCUM` | `8` | Quality samples per virtual accumulation window. |
| `MO_DPO_INDEX_JSON` | unset | Optional precomputed objective index file. |

Recommended example:

```bash
MO_DPO_QUALITY_LAMBDA=0.5 \
MO_DPO_USE_MARGIN=true \
MO_DPO_MARGIN_MODE=linear \
MO_DPO_TRUTH_MARGIN_SCALE=2.0 \
MO_DPO_QUALITY_MARGIN_SCALE=1.0 \
MO_DPO_MARGIN_MAX=1.0 \
MO_DPO_USE_OBJECTIVE_SAMPLER=true \
MO_DPO_TRUTH_PER_ACCUM=8 \
MO_DPO_QUALITY_PER_ACCUM=8 \
swift rlhf --rlhf_type objective_margin_dpo ...
```

## Notes on objective weighting

In `convex` mode, the effective loss is:

```text
(1 - MO_DPO_QUALITY_LAMBDA) * truth_loss
+ MO_DPO_QUALITY_LAMBDA * quality_loss
```

Because `MO_DPO_QUALITY_LAMBDA` defaults to `MO_DPO_QUALITY_WEIGHT`, explicitly
set one of them for your experiment. By default, `MO_DPO_QUALITY_WEIGHT=0.5`
gives `0.5 * truth_loss + 0.5 * quality_loss` in default `convex` mode,
matching the paper's `$\lambda=0.5$` setting.

## Paper hyperparameter mapping

The training hyperparameters reported in the paper map to this release as
follows:

| Paper symbol / setting | Release location |
|---|---|
| `$\lambda$` | `MO_DPO_QUALITY_LAMBDA`; defaults to `0.5`. In default `convex` mode, the loss is `(1 - lambda) * truth_loss + lambda * quality_loss`. |
| DPO `$\beta$` | ms-swift `--beta`; use `0.1` to match the paper. |
| Per-device train batch size | ms-swift `--per_device_train_batch_size`; use `1` to match the paper. |
| Gradient accumulation steps | ms-swift `--gradient_accumulation_steps`; use `16` to match the paper. |
| Learning rate | ms-swift `--learning_rate`; use `1e-4` for LoRA training as in the paper. |
| LoRA rank / alpha / target modules | ms-swift LoRA arguments; use rank `32`, alpha `32`, and `all-linear` target modules as in the paper. |

The paper's `$\gamma_{\mathrm{rel}}$`, `$\alpha_{\mathrm{rel}}$`,
`$\gamma_{\mathrm{gran}}$`, `$\alpha_{\mathrm{gran}}$`, and pair filtering
threshold `$\tau$` belong to the pair construction / scoring stage. They are not
runtime environment variables of this trainer patch. This trainer consumes the
resulting pair metadata, especially `objective_id`, `score_diff`, and optionally
`margin`.

If the dataset already contains a `margin` field and `MO_DPO_MARGIN_SOURCE` is
`auto` or `margin`, the trainer uses that precomputed margin first. Otherwise,
it falls back to computing a runtime margin from `score_diff` according to
`MO_DPO_MARGIN_MODE`, `MO_DPO_TRUTH_MARGIN_SCALE`, and
`MO_DPO_QUALITY_MARGIN_SCALE`.

## Notes on the sampler

The objective interleaving sampler produces a single index stream with a stable
long-run objective ratio. With DDP, the stream is sharded by rank. This does not
guarantee exact per-rank composition inside every gradient accumulation window.
For exact per-rank windows, use one rank or choose accumulation/window settings
that are compatible with your world size.