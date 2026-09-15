"""Train and evaluate the H local candidate classifier on a frozen D1 model."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from local_refiner_config import (
    DEFAULT_CONFIG,
    EXPECTED_REFINER_PARAMETERS,
    SOURCE_ROOT,
    json_value,
    load_runtime,
    write_json,
)
from local_refiner_reporting import benchmark_fp16, complexity_report, gate_result
from local_refiner_training import (
    CandidateCropDataset,
    append_results,
    checkpoint_metadata,
    generate_candidates,
    run_base_calibration,
    save_epoch_checkpoint,
    train_one_epoch,
    validate_refiner,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Override the frozen D1 best.pt path")
    parser.add_argument("--data", help="Override the data YAML path")
    parser.add_argument("--run-name", help="Override the output Run ID")
    parser.add_argument(
        "--preflight1",
        action="store_true",
        help="Run one refiner epoch under a timestamped preflight Run ID",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration only")
    return parser.parse_args()


def execute(runtime: dict[str, Any]) -> None:
    """Execute the complete H training, validation, and gate workflow."""
    sys.path.insert(0, str(SOURCE_ROOT))
    from evaluate_local_refiner import evaluate_original_images
    from ultralytics import YOLO, __version__
    from ultralytics.models.yolo.segment import load_local_refinement_checkpoint
    from ultralytics.nn.modules import LocalCropRefiner
    from ultralytics.utils.torch_utils import init_seeds, select_device

    train = runtime["train"]
    refiner_config = runtime["refiner"]
    seed = int(train["seed"])
    init_seeds(seed, deterministic=bool(train.get("deterministic", True)))
    device = select_device(str(train["device"]))
    runtime["run_dir"].mkdir(parents=True, exist_ok=False)
    (runtime["run_dir"] / "weights").mkdir()
    shutil.copy2(runtime["config_path"], runtime["run_dir"] / "experiment.yaml")
    write_json(
        runtime["run_dir"] / "runtime.json",
        {
            **runtime,
            "ultralytics": __version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    )

    base = YOLO(runtime["model"])
    base.model.eval()
    for parameter in base.model.parameters():
        parameter.requires_grad_(False)
    backbone_count = int(refiner_config.get("backbone_layers", 5))
    refiner = LocalCropRefiner(
        base.model.model[:backbone_count],
        in_channels=int(refiner_config.get("in_channels", 512)),
        num_classes=int(refiner_config.get("num_classes", 3)),
    )
    refiner_parameters = sum(parameter.numel() for parameter in refiner.parameters())
    if refiner_parameters != EXPECTED_REFINER_PARAMETERS:
        raise RuntimeError(
            f"unexpected refiner size: expected {EXPECTED_REFINER_PARAMETERS}, got {refiner_parameters}"
        )

    run_base_calibration(runtime, base.model)
    candidate_summary = generate_candidates(runtime, base.model)
    write_json(runtime["run_dir"] / "candidate_generation.json", candidate_summary)
    dataset = CandidateCropDataset(
        runtime["run_dir"] / "candidates",
        flipud=float(train.get("flipud", 0.5)),
        fliplr=float(train.get("fliplr", 0.5)),
    )

    refiner.to(device)
    optimizer_name = str(train.get("optimizer", "AdamW"))
    if optimizer_name != "AdamW":
        raise ValueError(f"H requires optimizer=AdamW, got {optimizer_name}")
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(train["lr0"]),
        weight_decay=float(train["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_fitness = float("-inf")
    results_path = runtime["run_dir"] / "results.csv"
    epochs = int(train["epochs"])
    for epoch in range(epochs):
        loss, accuracy, samples = train_one_epoch(
            refiner,
            dataset,
            optimizer,
            criterion,
            scaler,
            device,
            batch_size=int(train["batch"]),
            workers=int(train.get("workers", 8)),
            seed=seed,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )
        metrics, diagnostic_rate = validate_refiner(
            runtime,
            base.model,
            refiner,
            runtime["run_dir"] / "validation" / f"epoch_{epoch + 1:03d}",
        )
        results = metrics["results"]
        record = {
            "epoch": epoch + 1,
            "train/loss": loss,
            "train/accuracy": accuracy,
            "train/samples": samples,
            "lr/pg0": optimizer.param_groups[0]["lr"],
            **{key: float(value) for key, value in results.items()},
            "metrics/mAP75(M)": metrics["mask_ap75"],
            "diagnostic/low_overlap_high_confidence_rate": diagnostic_rate,
        }
        append_results(results_path, record)
        metadata = checkpoint_metadata(runtime, epoch + 1, loss, accuracy, metrics, diagnostic_rate)
        save_epoch_checkpoint(runtime["run_dir"] / "weights" / "last.pt", base.model, refiner, metadata)
        fitness = float(results["fitness"])
        if fitness > best_fitness:
            best_fitness = fitness
            save_epoch_checkpoint(runtime["run_dir"] / "weights" / "best.pt", base.model, refiner, metadata)
        print(
            f"[EPOCH {epoch + 1:02d}/{epochs}] loss={loss:.5f} accuracy={accuracy:.5f} "
            f"mask_map={float(results['metrics/mAP50-95(M)']):.5f} fitness={fitness:.5f}"
        )

    best_path = runtime["run_dir"] / "weights" / "best.pt"
    best_base, best_refiner, best_metadata = load_local_refinement_checkpoint(best_path)
    final_metrics, official_diagnostic_rate = validate_refiner(
        runtime,
        best_base,
        best_refiner,
        runtime["run_dir"] / "final_validation",
        plots=True,
    )
    write_json(
        runtime["run_dir"] / "best_metrics.json",
        {
            "selected_epoch": best_metadata["epoch"],
            "selection_metric": "official fitness = Box mAP50-95 + Mask mAP50-95",
            "metrics": final_metrics,
        },
    )
    original_evaluation = evaluate_original_images(
        checkpoint=best_path,
        data_yaml=runtime["data"],
        output_dir=runtime["run_dir"],
        device=str(train["device"]),
        imgsz=int(train["imgsz"]),
        crop_size=int(refiner_config["crop_size"]),
        score_batch_size=int(refiner_config["score_batch_size"]),
    )
    gate = gate_result(final_metrics, original_evaluation, official_diagnostic_rate)
    write_json(runtime["run_dir"] / "gate.json", gate)
    write_json(
        runtime["run_dir"] / "mechanism_diagnostics.json",
        {
            "reference_d1_rate": 0.2141,
            "official_val_rate": official_diagnostic_rate,
            "original_image_val": original_evaluation["low_overlap_high_confidence"],
        },
    )
    fp16 = benchmark_fp16(
        best_path,
        runtime["data"],
        str(train["device"]),
        int(train["imgsz"]),
        int(refiner_config["crop_size"]),
        int(refiner_config["score_batch_size"]),
    )
    report = complexity_report(
        best_base,
        best_refiner,
        int(train["imgsz"]),
        int(refiner_config["crop_size"]),
        original_evaluation,
        fp16,
    )
    write_json(runtime["run_dir"] / "complexity.json", report)
    print(f"[DONE] {runtime['run_dir']}")
    print(f"[GATE] {'PASS' if gate['passed'] else 'FAIL'}: {gate['next_action']}")


def main() -> None:
    """Run H from YAML configuration."""
    cli_args = parse_args()
    runtime = load_runtime(cli_args)
    print(yaml.safe_dump(json_value(runtime), allow_unicode=True, sort_keys=False).rstrip())
    if cli_args.dry_run:
        print("[DRY RUN] Configuration resolved; no Run directory was created.")
        return
    execute(runtime)


if __name__ == "__main__":
    main()
