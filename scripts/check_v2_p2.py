"""V2-P2 no-training gate checks.

This script never calls the Ultralytics training API, never creates an optimizer and never updates model weights.
It verifies model topology, tensor shapes, semantic pretrained-weight transfer, parameter count and FLOPs before the
user manually starts any smoke or formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = REPO_ROOT / "ultralytics-main" / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-p2-seg.yaml"
BASELINE_WEIGHTS = REPO_ROOT / "ultralytics-main" / "yolo26m-seg.pt"
PROFILE_PATH = REPO_ROOT / "experiments" / "yolo26m_seg_baseline_train.yaml"
EXPECTED_STRIDES = (4.0, 8.0, 16.0, 32.0)
EXPECTED_HEAD = "Segment26P2"
EXPECTED_NM = 32


def parse_args() -> argparse.Namespace:
    """Parse no-training check options."""
    parser = argparse.ArgumentParser(description="检查 V2-P2 模型，不启动训练。")
    parser.add_argument("--imgsz", type=int, default=256, help="随机前向检查尺寸；默认 256，必须能被 32 整除。")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    """Raise a readable gate failure."""
    if not condition:
        raise RuntimeError(message)


def module_hash(*modules: torch.nn.Module) -> str:
    """Hash module state tensors to prove that baseline loading did not overwrite new P2 weights."""
    digest = hashlib.sha256()
    for module in modules:
        for key, tensor in sorted(module.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def assert_module_equal(target: torch.nn.Module, source: torch.nn.Module, label: str) -> int:
    """Assert exact tensor equality for a semantically matched source/target module pair."""
    target_state = target.state_dict()
    source_state = source.state_dict()
    require(set(target_state) == set(source_state), f"{label} state keys do not match.")
    for key, source_tensor in source_state.items():
        target_tensor = target_state[key]
        require(
            target_tensor.shape == source_tensor.shape and torch.equal(target_tensor, source_tensor),
            f"{label}.{key} was not transferred exactly.",
        )
    return len(source_state)


def synthetic_loss_forward(initialized_v2, imgsz: int) -> dict:
    """Rebuild the two-class trainer model and calculate one synthetic batch without backward or an optimizer."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel

    with PROFILE_PATH.open("r", encoding="utf-8") as file:
        train_args = yaml.safe_load(file)["train"]

    # This mirrors the model rebuild performed by SegmentationTrainer after it reads dataset nc=2.
    model = SegmentationModel(initialized_v2.model.yaml, nc=2, verbose=False)
    model.load(initialized_v2.model, verbose=False)
    model.args = get_cfg(overrides=train_args)
    model.names = {0: "Rice leaffolder", 1: "Rice stemborers"}
    require(model.training and model.model[-1].training, "Synthetic-loss model must be in its default training mode.")

    mask_size = imgsz // 4
    masks = torch.zeros(2, mask_size, mask_size)
    masks[0, int(0.31 * mask_size) : int(0.66 * mask_size), int(0.28 * mask_size) : int(0.61 * mask_size)] = 1
    masks[1, int(0.25 * mask_size) : int(0.70 * mask_size), int(0.39 * mask_size) : int(0.75 * mask_size)] = 1
    sem_masks = torch.zeros(2, mask_size, mask_size)
    sem_masks[1][masks[1] > 0] = 1
    torch.manual_seed(0)
    batch = {
        "img": torch.rand(2, 3, imgsz, imgsz),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.445, 0.484, 0.328, 0.344], [0.570, 0.477, 0.359, 0.453]]),
        "masks": masks,
        "sem_masks": sem_masks,
    }

    # Deliberately no optimizer, no backward(), and no high-level training API.
    with torch.no_grad():
        loss_vector, loss_items = model(batch)

    require(torch.isfinite(loss_vector).all().item(), f"Synthetic total loss contains NaN/Inf: {loss_vector}.")
    require(torch.isfinite(loss_items).all().item(), f"Synthetic loss items contain NaN/Inf: {loss_items}.")
    require(loss_items.numel() == 5, f"Expected five segmentation loss items, got {loss_items.numel()}.")
    require(loss_items[1].item() > 0, f"Mask loss path was not active: {loss_items}.")
    require(loss_items[4].item() > 0, f"Semantic-mask loss path was not active: {loss_items}.")

    return {
        "status": "PASS - forward loss only; no optimizer/backward/training API",
        "criterion": model.criterion.__class__.__name__,
        "loss_vector": [round(float(value), 6) for value in loss_vector],
        "loss_sum": round(float(loss_vector.sum()), 6),
        "loss_items_box_seg_cls_dfl_sem": [round(float(value), 6) for value in loss_items],
        "trainer_rebuild_transfer": model.model[-1].pretrained_transfer_report,
    }


def main() -> None:
    """Run all V2-P2 checks without invoking training."""
    args = parse_args()
    require(args.imgsz > 0 and args.imgsz % 32 == 0, "--imgsz must be a positive multiple of 32.")
    require(MODEL_YAML.is_file(), f"Missing model YAML: {MODEL_YAML}")
    require(BASELINE_WEIGHTS.is_file(), f"Missing baseline weights: {BASELINE_WEIGHTS}")
    require(PROFILE_PATH.is_file(), f"Missing locked training profile: {PROFILE_PATH}")

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    v2 = YOLO(str(MODEL_YAML))
    net = v2.model
    head = net.model[-1]

    require(v2.task == "segment", f"Expected task=segment, got {v2.task}.")
    require(head.__class__.__name__ == EXPECTED_HEAD, f"Expected {EXPECTED_HEAD}, got {head.__class__.__name__}.")
    require(tuple(float(x) for x in head.stride.tolist()) == EXPECTED_STRIDES, f"Unexpected strides: {head.stride}.")
    require(tuple(head.proto_input_indices) == (1, 2, 3), "Proto must receive P3/P4/P5 only.")
    require(net.yaml.get("scale") == "m", f"Expected m scale, got {net.yaml.get('scale')}.")

    # Record every new P2 parameter before loading the baseline checkpoint.
    p2_hash_before = module_hash(net.model[23], net.model[24], net.model[25], head.cv2[0], head.cv3[0], head.cv4[0])
    if head.end2end:
        p2_hash_before = module_hash(
            net.model[23],
            net.model[24],
            net.model[25],
            head.cv2[0],
            head.cv3[0],
            head.cv4[0],
            head.one2one_cv2[0],
            head.one2one_cv3[0],
            head.one2one_cv4[0],
        )

    v2.load(str(BASELINE_WEIGHTS))
    report = head.pretrained_transfer_report
    require(report is not None, "Missing semantic pretrained-transfer report.")
    require(report.get("mode") == "baseline-segment26-to-p2", f"Unexpected transfer mode: {report}")

    baseline = YOLO(str(BASELINE_WEIGHTS))
    baseline_net = baseline.model
    baseline_head = baseline_net.model[-1]

    transferred_checks = 0
    # Layers 0-22 are intentionally identical to the baseline model.
    for layer_idx in range(23):
        transferred_checks += assert_module_equal(
            net.model[layer_idx],
            baseline_net.model[layer_idx],
            f"model layer {layer_idx}",
        )

    # Baseline P3/P4/P5 branches map to V2 target indices 1/2/3; index 0 is the new P2 branch.
    branch_names = ("cv2", "cv3", "cv4")
    if head.end2end:
        branch_names += ("one2one_cv2", "one2one_cv3", "one2one_cv4")
    for branch_name in branch_names:
        target_branches = getattr(head, branch_name)
        source_branches = getattr(baseline_head, branch_name)
        for source_idx, level in enumerate(("P3", "P4", "P5")):
            transferred_checks += assert_module_equal(
                target_branches[source_idx + 1],
                source_branches[source_idx],
                f"{branch_name}.{level}",
            )
    transferred_checks += assert_module_equal(head.proto, baseline_head.proto, "Proto26(P3/P4/P5)")

    p2_modules = [net.model[23], net.model[24], net.model[25], head.cv2[0], head.cv3[0], head.cv4[0]]
    if head.end2end:
        p2_modules.extend((head.one2one_cv2[0], head.one2one_cv3[0], head.one2one_cv4[0]))
    p2_hash_after = module_hash(*p2_modules)
    require(p2_hash_after == p2_hash_before, "Baseline loading unexpectedly overwrote new P2 parameters.")

    head_inputs: list[tuple[int, ...]] = []
    proto_inputs: list[tuple[int, ...]] = []

    def record_head_inputs(_module, inputs):
        head_inputs.extend(tuple(feature.shape) for feature in inputs[0])

    def record_proto_inputs(_module, inputs):
        proto_inputs.extend(tuple(feature.shape) for feature in inputs[0])

    head_hook = head.register_forward_pre_hook(record_head_inputs)
    proto_hook = head.proto.register_forward_pre_hook(record_proto_inputs)
    net.eval()
    with torch.inference_mode():
        output = net(torch.zeros(1, 3, args.imgsz, args.imgsz))
    head_hook.remove()
    proto_hook.remove()

    expected_hw = (
        args.imgsz // 4,
        args.imgsz // 8,
        args.imgsz // 16,
        args.imgsz // 32,
    )
    require([shape[-1] for shape in head_inputs] == list(expected_hw), f"Unexpected head inputs: {head_inputs}")
    require([shape[-1] for shape in proto_inputs] == list(expected_hw[1:]), f"Unexpected proto inputs: {proto_inputs}")

    detections, proto = output[0]
    raw_preds = output[1]
    require(tuple(detections.shape) == (1, 300, 6 + EXPECTED_NM), f"Unexpected detections: {detections.shape}")
    require(
        tuple(proto.shape) == (1, EXPECTED_NM, args.imgsz // 4, args.imgsz // 4),
        f"Proto resolution changed unexpectedly: {proto.shape}",
    )
    coefficient = raw_preds["one2many"]["mask_coefficient"]
    expected_candidates = sum(size * size for size in expected_hw)
    require(
        tuple(coefficient.shape) == (1, EXPECTED_NM, expected_candidates),
        f"Unexpected mask coefficient shape: {coefficient.shape}",
    )
    require(torch.isfinite(detections).all().item(), "Detection output contains NaN/Inf.")
    require(torch.isfinite(proto).all().item(), "Proto output contains NaN/Inf.")
    require(torch.isfinite(coefficient).all().item(), "Mask coefficients contain NaN/Inf.")

    v2_params = sum(parameter.numel() for parameter in net.parameters())
    baseline_params = sum(parameter.numel() for parameter in baseline_net.parameters())
    v2_flops = float(get_flops(net, imgsz=640))
    baseline_flops = float(get_flops(baseline_net, imgsz=640))
    loss_forward = synthetic_loss_forward(v2, args.imgsz)

    summary = {
        "status": "PASS - no training executed",
        "task": v2.task,
        "head": head.__class__.__name__,
        "scale": net.yaml.get("scale"),
        "strides": list(EXPECTED_STRIDES),
        "head_inputs_at_check_size": head_inputs,
        "proto_inputs_at_check_size": proto_inputs,
        "proto_output": list(proto.shape),
        "mask_coefficient": list(coefficient.shape),
        "semantic_transfer_report": report,
        "verified_transferred_tensors": transferred_checks,
        "new_p2_weights_unchanged_by_baseline_load": True,
        "synthetic_loss_forward": loss_forward,
        "params": {
            "baseline": baseline_params,
            "v2_p2": v2_params,
            "delta": v2_params - baseline_params,
        },
        "gflops_640": {
            "baseline": round(baseline_flops, 3),
            "v2_p2": round(v2_flops, 3),
            "delta": round(v2_flops - baseline_flops, 3),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
