"""Acceptance, complexity, and latency reporting for experiment H."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn


def gate_result(
    final_metrics: dict[str, Any],
    original_evaluation: dict[str, Any],
    official_diagnostic_rate: float,
) -> dict[str, Any]:
    """Apply the preregistered H acceptance rules without relaxing thresholds."""
    overall = final_metrics["results"]
    per_class = final_metrics["mask_per_class_ap50_95"]
    original_mask = original_evaluation["metrics"]["segm"]["overall"]["all"]["AP50_95"]
    original_diagnostic = original_evaluation["low_overlap_high_confidence"]
    conditions = {
        "official_mask_map_at_least_0_37827": float(overall["metrics/mAP50-95(M)"]) >= 0.37827,
        "original_mask_map_above_0_34814": float(original_mask) > 0.34814,
        "rice_leaffolder_drop_at_most_0_005": float(per_class[0]) >= 0.285,
        "rice_stemborers_parent_drop_at_most_0_005": float(per_class[1]) >= 0.453,
        "rice_stemborers_baseline_drop_at_most_0_005": float(per_class[1]) >= 0.456,
        "original_low_overlap_rate_below_0_2141": float(original_diagnostic["rate"]) < 0.2141,
    }
    passed = all(conditions.values())
    return {
        "passed": passed,
        "conditions": conditions,
        "observed": {
            "official_mask_map": float(overall["metrics/mAP50-95(M)"]),
            "official_mask_ap75": float(final_metrics["mask_ap75"]),
            "official_mask_per_class_ap50_95": per_class,
            "original_mask_map": float(original_mask),
            "official_low_overlap_high_confidence_rate": official_diagnostic_rate,
            "original_low_overlap_high_confidence": original_diagnostic,
        },
        "next_action": (
            "run seeds 2 and 3, then Baseline+H attribution if the three-seed rule passes"
            if passed
            else "do not use D1 alone; create the separate I dual-prototype branch"
        ),
    }


def benchmark_fp16(
    checkpoint: Path,
    data_yaml: Path,
    device: str,
    imgsz: int,
    crop_size: int,
    score_batch_size: int,
) -> dict[str, Any]:
    """Measure batch-1 FP16 latency and peak memory on up to 50 Val images."""
    if not torch.cuda.is_available() or str(device).lower() == "cpu":
        return {"available": False, "reason": "CUDA is required for the preregistered FP16 benchmark"}

    from local_refiner_coco import resolve_data_paths
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import LocalRefinementPredictor, load_local_refinement_checkpoint

    images, _, _ = resolve_data_paths(data_yaml)
    images = images[:50]
    _, refiner, _ = load_local_refinement_checkpoint(checkpoint)
    model = YOLO(checkpoint)
    predictor = partial(
        LocalRefinementPredictor,
        refiner=refiner,
        crop_size=crop_size,
        score_batch_size=score_batch_size,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    speeds: list[dict[str, float]] = []
    results = model.predict(
        source=[str(path) for path in images],
        stream=True,
        predictor=predictor,
        imgsz=imgsz,
        rect=False,
        batch=1,
        conf=0.001,
        iou=0.7,
        max_det=300,
        retina_masks=True,
        quantize=16,
        device=device,
        save=False,
        verbose=False,
    )
    for result in results:
        speeds.append({key: float(value) for key, value in result.speed.items()})
    measured = speeds[5:] if len(speeds) > 5 else speeds
    means = {key: sum(speed[key] for speed in measured) / len(measured) for key in measured[0]}
    return {
        "available": True,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "images": len(images),
        "warmup_images_excluded": len(speeds) - len(measured),
        "mean_ms": means,
        "mean_total_ms": sum(means.values()),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def complexity_report(
    base_model: nn.Module,
    refiner: nn.Module,
    imgsz: int,
    crop_size: int,
    original_evaluation: dict[str, Any],
    fp16_benchmark: dict[str, Any],
) -> dict[str, Any]:
    """Record static model size and dynamic per-candidate compute."""
    from ultralytics.utils.torch_utils import get_flops

    base_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    refiner_parameters = sum(parameter.numel() for parameter in refiner.parameters())
    base_gflops = float(get_flops(base_model.float().cpu(), imgsz=imgsz))
    refiner_gflops = float(get_flops(refiner.float().cpu(), imgsz=crop_size))
    if base_gflops <= 0 or refiner_gflops <= 0:
        raise RuntimeError("GFLOPs calculation failed; verify that ultralytics-thop is installed")
    image_count = int(original_evaluation["protocol"]["images"])
    mean_candidates = int(original_evaluation["prediction_count"]) / image_count
    return {
        "base_parameters": base_parameters,
        "refiner_parameters": refiner_parameters,
        "combined_parameters": base_parameters + refiner_parameters,
        "base_gflops_per_image": base_gflops,
        "refiner_gflops_per_candidate": refiner_gflops,
        "mean_candidates_per_val_image_at_conf_0_001": mean_candidates,
        "estimated_mean_combined_gflops": base_gflops + refiner_gflops * mean_candidates,
        "batch1_fp16": fp16_benchmark,
    }
