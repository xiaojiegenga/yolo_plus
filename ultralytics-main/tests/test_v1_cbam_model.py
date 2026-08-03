"""Focused structural tests for the independent V1-CBAM-b4 experiment."""

from pathlib import Path

import torch

from ultralytics.nn.modules import C3k2, C3k2CBAM, Segment26
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
V1_YAML = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-cbam-seg.yaml"


def test_c3k2_cbam_preserves_original_state_keys_and_shape():
    """The wrapper must preserve the original C3k2 path and only append CBAM tensors."""
    baseline = C3k2(64, 128, n=1, c3k=True, e=0.25)
    improved = C3k2CBAM(64, 128, n=1, c3k=True, e=0.25)

    baseline_state = baseline.state_dict()
    improved_state = improved.state_dict()
    assert set(baseline_state).issubset(improved_state)
    assert all(baseline_state[key].shape == improved_state[key].shape for key in baseline_state)
    assert all(key.startswith("cbam.") for key in set(improved_state).difference(baseline_state))

    output = improved(torch.zeros(1, 64, 16, 16))
    assert output.shape == (1, 128, 16, 16)


def test_v1_yaml_builds_four_backbone_cbam_blocks_and_segment_head():
    """The V1 YAML must remain an m-scale P3/P4/P5 segmentation model."""
    model = DetectionModel(V1_YAML, verbose=False)
    cbam_indices = [index for index, layer in enumerate(model.model) if isinstance(layer, C3k2CBAM)]

    assert model.yaml["scale"] == "m"
    assert cbam_indices == [2, 4, 6, 8]
    assert isinstance(model.model[-1], Segment26)
    assert model.model[-1].f == [16, 19, 22]

    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, 64, 64))


if __name__ == "__main__":
    test_c3k2_cbam_preserves_original_state_keys_and_shape()
    test_v1_yaml_builds_four_backbone_cbam_blocks_and_segment_head()
    print("V1-CBAM structural tests passed.")
