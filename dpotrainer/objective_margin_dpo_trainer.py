# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-objective margin-DPO trainer.

This trainer is intentionally a small extension on top of the local ms-swift
DPOTrainer:

* reuse DPOTrainer.concatenated_forward for policy/ref log-prob computation;
* pop objective-specific metadata before model forward;
* compute a sigmoid margin-DPO per-pair loss;
* reduce losses with objective-aware weighting, convex by default;
* optionally use an accumulation-aware sampler that works with micro-batch size 1.

Expected dataset fields:
    objective_id: 1 for truth, 2 for quality_f1
    score_diff: metric gap used by runtime margin computation

Environment variables are used deliberately so the script can work even before
the custom options are registered in ms-swift's CLI dataclasses.
"""

from __future__ import annotations

import math
import os
import json
import random
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from swift.utils import get_logger
from .dpo_trainer import DPOTrainer


logger = get_logger()

OBJECTIVE_TRUTH = 1
OBJECTIVE_QUALITY_F1 = 2
VALID_MARGIN_MODES = {"linear", "sqrt", "log", "constant", "none", "off", "false"}


def _sample_get(sample: Any, key: str, default=None):
    """Read metadata from either top-level sample fields or ms-swift _extra_kwargs."""
    if isinstance(sample, dict):
        if key in sample:
            return sample[key]
        extra = sample.get("_extra_kwargs")
        if isinstance(extra, dict) and key in extra:
            return extra[key]
    return default


def _pop_with_extra(inputs: Dict, key: str, default=None):
    """Pop a field from batch, falling back to _extra_kwargs when present."""
    if key in inputs:
        return inputs.pop(key)
    extra = inputs.get("_extra_kwargs")
    if isinstance(extra, dict) and key in extra:
        return extra.pop(key)
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", "none"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


class ObjectiveInterleavingSampler(Sampler[int]):
    """Sampler that balances objectives across gradient accumulation windows.

    It yields a single stream of indices, so it is compatible with
    per_device_train_batch_size=1. Within each virtual accumulation window it
    emits truth_per_accum truth samples and quality_per_accum quality samples.
    Minority objectives are oversampled with replacement when needed.
    """

    def __init__(
        self,
        dataset,
        truth_per_accum: int,
        quality_per_accum: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        strict: bool = True,
    ) -> None:
        if truth_per_accum <= 0 or quality_per_accum <= 0:
            raise ValueError(
                f"truth_per_accum and quality_per_accum must be positive, got "
                f"{truth_per_accum}, {quality_per_accum}"
            )
        self.dataset = dataset
        self.truth_per_accum = truth_per_accum
        self.quality_per_accum = quality_per_accum
        self.window_size = truth_per_accum + quality_per_accum
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.strict = strict
        self.epoch = 0

        self.truth_indices: List[int] = []
        self.quality_indices: List[int] = []
        self.other_indices: List[int] = []

        if not self._load_indices_from_json(len(dataset)):
            objective_ids = self._try_get_objective_column(dataset)
            if objective_ids is not None:
                logger.info("[ObjectiveInterleavingSampler] Using dataset objective_id column for indices.")
                self._scan_objective_values(objective_ids)
            else:
                logger.info("[ObjectiveInterleavingSampler] Scanning dataset for objective_id...")
                for i in range(len(dataset)):
                    if i > 0 and i % 10000 == 0:
                        logger.info(f"[ObjectiveInterleavingSampler] scanned {i}/{len(dataset)} samples...")
                    sample = dataset[i]
                    objective_id = _sample_get(sample, "objective_id", _sample_get(sample, "stage_type", None))
                    self._append_index(i, objective_id)

        logger.info(
            f"[ObjectiveInterleavingSampler] Found truth={len(self.truth_indices)}, "
            f"quality_f1={len(self.quality_indices)}, other={len(self.other_indices)}, "
            f"truth_per_accum={truth_per_accum}, quality_per_accum={quality_per_accum}, "
            f"rank={rank}, world_size={world_size}"
        )
        if strict and self.other_indices:
            raise ValueError(f"Found {len(self.other_indices)} samples with missing/unknown objective_id")
        if not self.truth_indices:
            raise ValueError("No truth objective samples found")
        if not self.quality_indices:
            raise ValueError("No quality_f1 objective samples found")

        n_truth_windows = math.ceil(len(self.truth_indices) / truth_per_accum)
        n_quality_windows = math.ceil(len(self.quality_indices) / quality_per_accum)
        self.num_windows = max(n_truth_windows, n_quality_windows)
        self.num_samples = self.num_windows * self.window_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _append_index(self, i: int, objective_id) -> None:
        try:
            objective_id = int(objective_id)
        except Exception:
            objective_id = None
        if objective_id == OBJECTIVE_TRUTH:
            self.truth_indices.append(i)
        elif objective_id == OBJECTIVE_QUALITY_F1:
            self.quality_indices.append(i)
        else:
            self.other_indices.append(i)

    def _scan_objective_values(self, objective_ids) -> None:
        for i, objective_id in enumerate(objective_ids):
            self._append_index(i, objective_id)

    def _load_indices_from_json(self, dataset_len: int) -> bool:
        index_json = os.environ.get("MO_DPO_INDEX_JSON")
        if not index_json:
            return False
        if not os.path.isfile(index_json):
            logger.warning(f"[ObjectiveInterleavingSampler] MO_DPO_INDEX_JSON not found: {index_json}; fallback to scan.")
            return False
        try:
            with open(index_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            num_pairs = int(data.get("num_pairs", -1))
            truth_indices = [int(x) for x in data.get("truth_indices", [])]
            quality_indices = [int(x) for x in data.get("quality_indices", [])]
        except Exception as exc:
            logger.warning(f"[ObjectiveInterleavingSampler] Failed to read MO_DPO_INDEX_JSON={index_json}: {exc}; fallback to scan.")
            return False
        if num_pairs != dataset_len:
            logger.warning(
                f"[ObjectiveInterleavingSampler] Index length mismatch: index num_pairs={num_pairs}, "
                f"dataset_len={dataset_len}; fallback to scan."
            )
            return False
        max_index = max(truth_indices + quality_indices) if truth_indices or quality_indices else -1
        if max_index >= dataset_len:
            logger.warning(
                f"[ObjectiveInterleavingSampler] Index contains out-of-range index {max_index} for dataset_len={dataset_len}; fallback to scan."
            )
            return False
        self.truth_indices = truth_indices
        self.quality_indices = quality_indices
        indexed = set(truth_indices) | set(quality_indices)
        if self.strict and len(indexed) != dataset_len:
            self.other_indices = [i for i in range(dataset_len) if i not in indexed]
        logger.info(
            f"[ObjectiveInterleavingSampler] Loaded objective indices from {index_json}: "
            f"truth={len(self.truth_indices)}, quality_f1={len(self.quality_indices)}, other={len(self.other_indices)}"
        )
        return True

    def _try_get_objective_column(self, dataset):
        for key in ("objective_id", "stage_type"):
            try:
                values = dataset[key]
            except Exception:
                continue
            if values is not None and len(values) == len(dataset):
                return values
        return None

    def _take(self, pool: List[int], start: int, count: int, rng: random.Random) -> List[int]:
        result = []
        for j in range(count):
            pos = start + j
            if pos < len(pool):
                result.append(pool[pos])
            else:
                result.append(rng.choice(pool))
        return result

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch + self.rank * 100_003)
        truth = self.truth_indices.copy()
        quality = self.quality_indices.copy()
        rng.shuffle(truth)
        rng.shuffle(quality)

        sequence: List[int] = []
        for window_idx in range(self.num_windows):
            window = []
            window.extend(self._take(truth, window_idx * self.truth_per_accum, self.truth_per_accum, rng))
            window.extend(self._take(quality, window_idx * self.quality_per_accum, self.quality_per_accum, rng))
            rng.shuffle(window)
            sequence.extend(window)

        # Shard the sample stream across DDP ranks. Each rank still sees a stable
        # long-run objective ratio. For exact per-rank windows, run with one rank
        # or choose accumulation sizes divisible by world_size.
        for idx in sequence[self.rank::self.world_size]:
            yield idx

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.world_size)


class ObjectiveMarginDPOTrainer(DPOTrainer):
    """DPOTrainer with objective-aware weighting and runtime margin computation."""

    def __init__(self, *args, **kwargs):
        self.quality_weight = _env_float("MO_DPO_QUALITY_WEIGHT", 0.5)
        self.quality_lambda = _env_float("MO_DPO_QUALITY_LAMBDA", self.quality_weight)
        self.objective_weight_mode = os.environ.get("MO_DPO_OBJECTIVE_WEIGHT_MODE", "convex").lower()
        self.use_objective_weight = _env_bool("MO_DPO_USE_OBJECTIVE_WEIGHT", True)
        self.use_margin = _env_bool("MO_DPO_USE_MARGIN", True)
        self.margin_source = os.environ.get("MO_DPO_MARGIN_SOURCE", "auto").lower()
        self.margin_beta_mode = os.environ.get("MO_DPO_MARGIN_BETA_MODE", "inside").lower()
        self.margin_mode = os.environ.get("MO_DPO_MARGIN_MODE", "linear").lower()
        self.truth_margin_scale = _env_float("MO_DPO_TRUTH_MARGIN_SCALE", 2.0)
        self.quality_margin_scale = _env_float("MO_DPO_QUALITY_MARGIN_SCALE", 1.0)
        self.margin_min = _env_float("MO_DPO_MARGIN_MIN", 0.0)
        self.margin_max = _env_float("MO_DPO_MARGIN_MAX", 1.0)
        self.strict_fields = _env_bool("MO_DPO_STRICT_FIELDS", True)
        self.use_objective_sampler = _env_bool("MO_DPO_USE_OBJECTIVE_SAMPLER", True)
        self.truth_per_accum = _env_int("MO_DPO_TRUTH_PER_ACCUM", 8)
        self.quality_per_accum = _env_int("MO_DPO_QUALITY_PER_ACCUM", 8)
        logger.info(
            "[ObjectiveMarginDPOTrainer] enabled: "
            f"quality_weight={self.quality_weight}, quality_lambda={self.quality_lambda}, "
            f"objective_weight_mode={self.objective_weight_mode}, use_objective_weight={self.use_objective_weight}, "
            f"use_margin={self.use_margin}, margin_mode={self.margin_mode}, "
            f"margin_source={self.margin_source}, margin_beta_mode={self.margin_beta_mode}, "
            f"truth_margin_scale={self.truth_margin_scale}, quality_margin_scale={self.quality_margin_scale}, "
            f"margin_min={self.margin_min}, margin_max={self.margin_max}, "
            f"use_objective_sampler={self.use_objective_sampler}, "
            f"truth_per_accum={self.truth_per_accum}, quality_per_accum={self.quality_per_accum}"
        )
        super().__init__(*args, **kwargs)
        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if loss_types != ["sigmoid"]:
            raise ValueError(
                f"ObjectiveMarginDPOTrainer currently supports only sigmoid loss, got loss_type={self.loss_type!r}"
            )
        if self.objective_weight_mode not in {"convex", "sum"}:
            raise ValueError("MO_DPO_OBJECTIVE_WEIGHT_MODE must be 'convex' or 'sum'")
        if self.objective_weight_mode == "convex" and not (0.0 <= self.quality_lambda <= 1.0):
            raise ValueError("MO_DPO_QUALITY_LAMBDA must be in [0, 1] when MO_DPO_OBJECTIVE_WEIGHT_MODE=convex")
        if self.margin_source not in {"auto", "margin", "score_diff"}:
            raise ValueError("MO_DPO_MARGIN_SOURCE must be 'auto', 'margin', or 'score_diff'")
        if self.margin_beta_mode not in {"inside", "outside"}:
            raise ValueError("MO_DPO_MARGIN_BETA_MODE must be 'inside' or 'outside'")
        if self.margin_mode not in VALID_MARGIN_MODES:
            raise ValueError(f"MO_DPO_MARGIN_MODE must be one of {sorted(VALID_MARGIN_MODES)}, got {self.margin_mode!r}")
        if self.objective_weight_mode == "convex":
            logger.info(
                "[ObjectiveMarginDPOTrainer] convex objective coefficients: "
                f"truth={1.0 - self.quality_lambda}, quality_f1={self.quality_lambda}"
            )
        else:
            logger.info(
                "[ObjectiveMarginDPOTrainer] sum objective coefficients: "
                f"truth=1.0, quality_f1={self.quality_weight}"
            )
        logger.info(
            f"[ObjectiveMarginDPOTrainer] ref_model_is_none={self.ref_model is None}, "
            f"has_null_ref_context={hasattr(self, 'null_ref_context')}"
        )
        self._objective_metrics = defaultdict(lambda: defaultdict(list))

    def _metric_store(self):
        """Return the metric buffer used by the active ms-swift/TRL version."""
        if hasattr(self, "_stored_metrics"):
            return self._stored_metrics
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        return self._metrics

    def _pop_objective_fields(self, inputs: Dict) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        objective_id = _pop_with_extra(inputs, "objective_id", None)
        _pop_with_extra(inputs, "objective_type", None)
        _pop_with_extra(inputs, "objective_score_metric", None)
        score_diff = _pop_with_extra(inputs, "score_diff", None)
        margin = _pop_with_extra(inputs, "margin", None)
        # Backward compatibility with stage fields.
        _pop_with_extra(inputs, "stage_name", None)
        if objective_id is None:
            objective_id = _pop_with_extra(inputs, "stage_type", None)
        else:
            _pop_with_extra(inputs, "stage_type", None)
        if isinstance(inputs.get("_extra_kwargs"), dict) and not inputs["_extra_kwargs"]:
            inputs.pop("_extra_kwargs", None)
        return objective_id, score_diff, margin

    def _to_pair_tensor(self, value, like: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=like.device, dtype=dtype)
        else:
            tensor = torch.tensor(value, device=like.device, dtype=dtype)
        tensor = tensor.flatten()
        if tensor.numel() == like.numel() * 2:
            tensor = tensor[: like.numel()]
        if tensor.numel() != like.numel():
            if tensor.numel() == 1:
                tensor = tensor.expand_as(like)
            else:
                raise ValueError(f"Objective field shape {tuple(tensor.shape)} does not match pair loss shape {tuple(like.shape)}")
        return tensor

    def _margin_transform(self, score_diff: torch.Tensor) -> torch.Tensor:
        score_diff = torch.clamp(score_diff, min=0.0)
        if self.margin_mode in {"none", "off", "false"}:
            return torch.zeros_like(score_diff)
        if self.margin_mode == "linear":
            return score_diff
        if self.margin_mode == "sqrt":
            return torch.sqrt(score_diff)
        if self.margin_mode == "log":
            return torch.log1p(score_diff)
        if self.margin_mode == "constant":
            return (score_diff > 0).to(score_diff.dtype)
        raise ValueError(f"Unsupported MO_DPO_MARGIN_MODE={self.margin_mode}")

    def _compute_margin(self, objective_id: Optional[torch.Tensor], score_diff, margin_raw, losses_like: torch.Tensor) -> torch.Tensor:
        if not self.use_margin:
            return torch.zeros_like(losses_like)
        if self.margin_source in {"auto", "margin"} and margin_raw is not None:
            margin = self._to_pair_tensor(margin_raw, losses_like, losses_like.dtype)
            assert margin is not None
            return torch.clamp(margin, min=self.margin_min, max=self.margin_max)
        if self.margin_source == "margin":
            raise ValueError("margin is required when MO_DPO_MARGIN_SOURCE=margin")
        if score_diff is None:
            if self.strict_fields:
                raise ValueError("score_diff is required when MO_DPO_USE_MARGIN=true")
            return torch.zeros_like(losses_like)
        diff = self._to_pair_tensor(score_diff, losses_like, losses_like.dtype)
        assert diff is not None
        base = self._margin_transform(diff)
        if objective_id is None:
            if self.strict_fields:
                raise ValueError("objective_id is required when MO_DPO_USE_MARGIN=true")
            scale = torch.ones_like(base)
        else:
            obj = self._to_pair_tensor(objective_id, losses_like, torch.long)
            assert obj is not None
            scale = torch.where(
                obj == OBJECTIVE_TRUTH,
                torch.full_like(base, self.truth_margin_scale),
                torch.where(obj == OBJECTIVE_QUALITY_F1, torch.full_like(base, self.quality_margin_scale), torch.ones_like(base)),
            )
            if self.strict_fields and torch.any((obj != OBJECTIVE_TRUTH) & (obj != OBJECTIVE_QUALITY_F1)):
                raise ValueError(f"Unknown objective_id values: {torch.unique(obj).detach().cpu().tolist()}")
        return torch.clamp(base * scale, min=self.margin_min, max=self.margin_max)

    def _reference_context(self, model):
        # TRL DPOTrainer exposes null_ref_context in many versions. Use it if
        # present so PEFT reference-adapter behavior remains intact.
        if hasattr(self, "null_ref_context"):
            return self.null_ref_context()
        if self.ref_model is None:
            raise RuntimeError(
                "ref_model is None but null_ref_context is unavailable; cannot safely compute DPO reference logps."
            )
        return nullcontext()

    def _compute_ref_outputs(self, model, inputs: Dict) -> Dict[str, torch.Tensor]:
        if self.precompute_ref_log_probs:
            return {
                "chosen_logps": inputs["ref_chosen_logps"],
                "rejected_logps": inputs["ref_rejected_logps"],
            }
        with torch.no_grad():
            if self.ref_model is None:
                with self._reference_context(model):
                    return self.concatenated_forward(model, inputs, is_ref_model=True)
            return self.concatenated_forward(self.ref_model, inputs, is_ref_model=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = inputs.copy()
        objective_id_raw, score_diff_raw, margin_raw = self._pop_objective_fields(inputs)

        if self.strict_fields and objective_id_raw is None:
            raise ValueError("objective_id/stage_type is required for ObjectiveMarginDPOTrainer")

        policy_outputs = self.concatenated_forward(model, inputs, is_ref_model=False)
        ref_outputs = self._compute_ref_outputs(model, inputs)

        chosen_logps = policy_outputs["chosen_logps"]
        rejected_logps = policy_outputs["rejected_logps"]
        if self.reference_free:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        else:
            ref_chosen_logps = ref_outputs["chosen_logps"].to(chosen_logps.device)
            ref_rejected_logps = ref_outputs["rejected_logps"].to(rejected_logps.device)

        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        margin = self._compute_margin(objective_id_raw, score_diff_raw, margin_raw, logits)
        if self.margin_beta_mode == "inside":
            per_pair_losses = -F.logsigmoid(self.beta * (logits - margin))
        else:
            per_pair_losses = -F.logsigmoid(self.beta * logits - margin)

        objective_id = self._to_pair_tensor(objective_id_raw, per_pair_losses, torch.long)
        mode = "train" if self.model.training else "eval"
        truth_mask = quality_mask = None

        if objective_id is None or not self.use_objective_weight:
            loss = per_pair_losses.mean()
            truth_loss = quality_loss = None
            truth_count = quality_count = 0
        else:
            truth_mask = objective_id == OBJECTIVE_TRUTH
            quality_mask = objective_id == OBJECTIVE_QUALITY_F1
            if self.strict_fields and torch.any(~(truth_mask | quality_mask)):
                raise ValueError(f"Unknown objective_id values: {torch.unique(objective_id).detach().cpu().tolist()}")
            truth_count = int(truth_mask.sum().item())
            quality_count = int(quality_mask.sum().item())
            truth_loss = per_pair_losses[truth_mask].mean() if truth_mask.any() else None
            quality_loss = per_pair_losses[quality_mask].mean() if quality_mask.any() else None
            pieces = []
            if truth_loss is not None:
                pieces.append((1.0 - self.quality_lambda) * truth_loss if self.objective_weight_mode == "convex" else truth_loss)
            if quality_loss is not None:
                quality_coef = self.quality_lambda if self.objective_weight_mode == "convex" else self.quality_weight
                pieces.append(quality_coef * quality_loss)
            if not pieces:
                loss = per_pair_losses.mean()
            else:
                loss = sum(pieces)

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        metrics = self._metric_store()
        metrics[mode]["rewards/chosen"].append(self.accelerator.gather(chosen_rewards).mean().item())
        metrics[mode]["rewards/rejected"].append(self.accelerator.gather(rejected_rewards).mean().item())
        metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather((chosen_rewards > rejected_rewards).float()).mean().item()
        )
        metrics[mode]["rewards/margins"].append(self.accelerator.gather(chosen_rewards - rejected_rewards).mean().item())
        metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps.detach()).mean().item())
        metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps.detach()).mean().item())

        self._objective_metrics[mode]["margin_mean"].append(self.accelerator.gather(margin.detach()).mean().item())
        self._objective_metrics[mode]["quality_weight"].append(self.quality_weight)
        self._objective_metrics[mode]["quality_lambda"].append(self.quality_lambda)
        self._objective_metrics[mode]["truth_count"].append(truth_count)
        self._objective_metrics[mode]["quality_count"].append(quality_count)
        if truth_loss is not None:
            self._objective_metrics[mode]["truth_loss"].append(truth_loss.detach().item())
            self._objective_metrics[mode]["truth_margin_mean"].append(margin[truth_mask].detach().mean().item())
        if quality_loss is not None:
            self._objective_metrics[mode]["quality_loss"].append(quality_loss.detach().item())
            self._objective_metrics[mode]["quality_margin_mean"].append(margin[quality_mask].detach().mean().item())
        score_diff_tensor = self._to_pair_tensor(score_diff_raw, per_pair_losses, per_pair_losses.dtype) if score_diff_raw is not None else None
        if score_diff_tensor is not None and truth_mask is not None and quality_mask is not None:
            if truth_mask.any():
                self._objective_metrics[mode]["truth_score_diff_mean"].append(score_diff_tensor[truth_mask].detach().mean().item())
            if quality_mask.any():
                self._objective_metrics[mode]["quality_score_diff_mean"].append(score_diff_tensor[quality_mask].detach().mean().item())

        if self.aux_loss_enabled and "aux_loss" in policy_outputs:
            loss = loss + self.aux_loss_coef * policy_outputs["aux_loss"]
        if self.args.rpo_alpha is not None and "nll_loss" in policy_outputs:
            loss = loss + self.args.rpo_alpha * policy_outputs["nll_loss"]
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps

        return (loss, policy_outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if not self.use_objective_sampler:
            return super().get_train_dataloader()
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        sampler = ObjectiveInterleavingSampler(
            self.train_dataset,
            truth_per_accum=self.truth_per_accum,
            quality_per_accum=self.quality_per_accum,
            seed=self.args.seed or 0,
            rank=rank,
            world_size=world_size,
            strict=self.strict_fields,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        if mode in self._objective_metrics:
            for key, values in self._objective_metrics[mode].items():
                if values:
                    logs[f"{mode}/objective_{key}"] = sum(values) / len(values)
            self._objective_metrics[mode].clear()
        super().log(logs, start_time)

    def training_step(self, model, inputs, *args, **kwargs):
        template = getattr(self, "template", None)
        if template is not None and hasattr(template, "forward_context"):
            context = template.forward_context(self.model, inputs)
        else:
            context = nullcontext()
        with context:
            return super().training_step(model, inputs, *args, **kwargs)