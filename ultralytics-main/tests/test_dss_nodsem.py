# Ultralytics AGPL-3.0 License - https://www.ultralytics.com/legal/license

"""Focused CPU tests for the nodsem ablation (official C3k2 restored on backbone layer 2).

Run from the repository root either way:

    python ultralytics-main/tests/test_dss_nodsem.py
    python -m pytest ultralytics-main/tests/test_dss_nodsem.py -q

This variant is YAML-only: layer 2 reverts to the official C3k2, so no DSEM tensors exist.
The head stays the full Segment26DSS (SFCM gate on, DPRM dual-resolution on, P2 injection on
-- it now taps the un-enhanced layer-2 output). Checks:
1. YAML parsing: layer 2 is a plain C3k2, layer 23 is Segment26DSS with both switches on;
2. no DSEM tensors in the state dict; shapes keep the stride-2 prototypes;
3. official ckpt transfer audit: 29 new tensors (the 15 DSEM ones of cmp1 are gone),
   only the 14 semseg tensors dropped from the checkpoint;
4. initial detection equivalence against the official yolo26m-seg model;
5. gradient flow into p2_proj / gate_scale / hi_out (everything but DSEM learns);
6. experiment config train block matches the stage-0 baseline recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # .../ultralytics-main
REPO_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from ultralytics.nn.modules import C3k2, Segment26DSS  # noqa: E402
from ultralytics.nn.tasks import SegmentationModel, yaml_model_load  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

NODSEM_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-nodsem-seg.yaml"
OFFICIAL_CKPT = Path(r"E:\study\graduate_sec\论文撰写\模型训练\.cache\yolo26m-seg.pt")
BASELINE_CFG = REPO_ROOT / "experiments" / "data-v2-abl-000-y26m-b16-s42.yaml"
NODSEM_CFG = REPO_ROOT / "experiments" / "data-v2-cmp2-nodsem-b16-s42.yaml"

CH = (256, 256, 512, 512)


def _feats(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 256, 160, 160),
        torch.randn(batch, 256, 80, 80),
        torch.randn(batch, 512, 40, 40),
        torch.randn(batch, 512, 20, 20),
    ]


def test_yaml_parses() -> None:
    """nodsem YAML restores the official C3k2 on layer 2 and keeps the full DSS head."""
    model = SegmentationModel(cfg=str(NODSEM_YAML), ch=3, nc=80, verbose=False)
    layer2 = model.model[2]
    assert type(layer2) is C3k2, f"layer 2 must be a plain C3k2, got {type(layer2).__name__}"
    head = model.model[23]
    assert type(head) is Segment26DSS, "layer 23 must stay the full Segment26DSS"
    assert head.sfcm_gate is True and head.dual_output is True, "gate and dual-resolution must stay on"
    keys = set(model.state_dict().keys())
    assert not any(k.startswith("model.2.dw.") or k.startswith("model.2.pw.") or k == "model.2.alpha" for k in keys), (
        "no DSEM tensors may exist in the nodsem model"
    )
    print(f"[yaml] layers={len(model.model)} params={sum(p.numel() for p in model.parameters())}")


def test_shapes_and_gradients() -> None:
    """Stride-2 prototypes stay; every remaining new component still learns."""
    torch.manual_seed(1)
    head = Segment26DSS(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    preds = head(_feats(2))
    proto, semantic = preds["proto"]
    assert proto.shape == (2, 32, 320, 320) and semantic.shape == (2, 2, 80, 80)
    (proto.sum() + semantic.sum()).backward()
    params = dict(head.named_parameters())
    for group in ("proto.p2_proj", "sem_trunk", "sem_head"):
        grads = [p.grad for n, p in params.items() if n.startswith(group)]
        assert all(g is not None for g in grads) and any(torch.any(g != 0) for g in grads), f"{group} must learn"
    assert head.gate_scale.grad is not None and torch.any(head.gate_scale.grad != 0)
    print("[gradients] p2_proj / trunk / head / gate all learning (DSEM absent by design)")


def test_official_transfer_and_equivalence() -> None:
    """29 new tensors (cmp1's 44 minus the 15 DSEM ones); initial detection equals the official model."""
    if not OFFICIAL_CKPT.exists():
        print(f"[transfer] SKIP: official checkpoint not found at {OFFICIAL_CKPT}")
        return
    torch.manual_seed(2)
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    csd = (ckpt["ema" if ckpt.get("ema") else "model"]).float().state_dict()

    nodsem = SegmentationModel(cfg=str(NODSEM_YAML), ch=3, nc=80, verbose=False)
    matched = intersect_dicts(csd, nodsem.state_dict())
    new_keys = set(nodsem.state_dict().keys()) - set(matched.keys())
    dropped = set(csd.keys()) - set(matched.keys())
    assert len(new_keys) == 29, f"expected 29 new tensors (44 - 15 DSEM), got {len(new_keys)}"
    prefixes = (
        "model.23.proto.p2_proj.",
        "model.23.proto.hi_up.",
        "model.23.proto.hi_cv.",
        "model.23.proto.hi_strip.",
        "model.23.proto.hi_out.",
        "model.23.sem_trunk.",
        "model.23.sem_head.",
        "model.23.sem_gate.",
        "model.23.gate_scale",
    )
    for key in sorted(new_keys):
        assert any(key.startswith(p) for p in prefixes), f"unexpected unloaded tensor: {key}"
    assert dropped and all("semseg" in k for k in dropped), f"unexpected dropped tensors: {sorted(dropped)[:4]}"
    print(f"[transfer] matched={len(matched)}/{len(nodsem.state_dict())}, new={len(new_keys)}, semseg dropped={len(dropped)}")

    nodsem.load_state_dict(matched, strict=False)
    official = SegmentationModel(cfg=yaml_model_load("yolo26m-seg.yaml"), ch=3, nc=80, verbose=False)
    official.load_state_dict(csd, strict=False)
    nodsem.eval(), official.eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        y_abl = nodsem(x)[0][0]
        y_official = official(x)[0][0]
    assert y_abl.shape == y_official.shape and torch.equal(y_abl, y_official), (
        "detection output must equal the official model at init"
    )
    print(f"[equivalence] detect={tuple(y_abl.shape)} equal=True")


def test_config_matches_baseline() -> None:
    """The nodsem experiment train block is identical to the stage-0 baseline recipe."""
    import yaml

    with BASELINE_CFG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with NODSEM_CFG.open("r", encoding="utf-8") as f:
        abl = yaml.safe_load(f)
    assert abl["data"] == base["data"]
    assert abl["train"] == base["train"], "train recipe must match the frozen baseline block"
    assert abl["model"].endswith("yolo26m-dss-nodsem-seg.yaml")
    print("[config] train block identical to 000 baseline")


def main() -> int:
    tests = [
        test_yaml_parses,
        test_shapes_and_gradients,
        test_official_transfer_and_equivalence,
        test_config_matches_baseline,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
