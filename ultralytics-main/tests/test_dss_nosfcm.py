# Ultralytics AGPL-3.0 License - https://www.ultralytics.com/legal/license

"""Focused CPU tests for the nosfcm ablation (Segment26DSSNoSFCM).

Run from the repository root either way:

    python ultralytics-main/tests/test_dss_nosfcm.py
    python -m pytest ultralytics-main/tests/test_dss_nosfcm.py -q

Checks:
1. registration + YAML parsing; layer 23 is the ablation head with the gate flag off;
2. the gate is truly bypassed: sem_gate weights cannot influence any output;
3. gate parameters receive no gradient while sem_trunk/sem_head do (aux supervision alive);
4. the training tuple still carries semantic logits (loss[4] slot stays fed);
5. the cmp1 checkpoint loads with strict weight parity (sfcm_gate is a class attribute, so
   checkpoints pickled before it existed resolve it from the class: Segment26DSS keeps gating ON,
   the ablation subclass keeps it OFF);
6. official checkpoint transfer audit is unchanged vs cmp1;
7. experiment config train block matches the stage-0 baseline recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # .../ultralytics-main
REPO_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from ultralytics.nn.modules import Segment26DSS, Segment26DSSNoSFCM  # noqa: E402
from ultralytics.nn.tasks import SegmentationModel  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

NOSFCM_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-nosfcm-seg.yaml"
DSS_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-seg.yaml"
OFFICIAL_CKPT = Path(r"E:\study\graduate_sec\论文撰写\模型训练\.cache\yolo26m-seg.pt")
CMP1_CKPT = REPO_ROOT / "runs" / "data-v2-cmp1-dss-b16-s42" / "weights" / "best.pt"
BASELINE_CFG = REPO_ROOT / "experiments" / "data-v2-abl-000-y26m-b16-s42.yaml"
NOSFCM_CFG = REPO_ROOT / "experiments" / "data-v2-cmp2-nosfcm-b16-s42.yaml"

CH = (256, 256, 512, 512)


def _feats(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 256, 160, 160),
        torch.randn(batch, 256, 80, 80),
        torch.randn(batch, 512, 40, 40),
        torch.randn(batch, 512, 20, 20),
    ]


def test_registration_and_flags() -> None:
    """The ablation subclass is registered and flips only the gate flag."""
    import ultralytics.nn.tasks as tasks

    assert hasattr(tasks, "Segment26DSSNoSFCM"), "Segment26DSSNoSFCM missing in ultralytics.nn.tasks"
    assert Segment26DSS.sfcm_gate is True, "cmp1 head must keep the gate on (class default)"
    assert Segment26DSSNoSFCM.sfcm_gate is False, "ablation head must disable the gate"


def test_yaml_parses() -> None:
    """yolo26m-dss-nosfcm-seg.yaml builds the ablation head with cmp1's layer layout."""
    model = SegmentationModel(cfg=str(NOSFCM_YAML), ch=3, nc=80, verbose=False)
    head = model.model[23]
    assert isinstance(head, Segment26DSSNoSFCM), "layer 23 must be Segment26DSSNoSFCM"
    assert head.sfcm_gate is False
    assert type(model.model[2]).__name__ == "C3k2DSEM", "DSEM must stay on layer 2"
    assert type(head.proto).__name__ == "Proto26DR", "DPRM must stay unchanged"
    print(f"[yaml] layers={len(model.model)} params={sum(p.numel() for p in model.parameters())}")


def test_gate_bypassed() -> None:
    """sem_gate weights cannot influence any forward output."""
    torch.manual_seed(1)
    head = Segment26DSSNoSFCM(nc=2, nm=32, npr=256, ch=CH)
    assert torch.equal(head.gate_scale, torch.zeros(CH[1])), "gate_scale must still init at zero"
    head.eval()
    feats = _feats(1)
    with torch.no_grad():
        out_a = head(feats)[0]
        head.sem_gate.weight.data.fill_(10.0)  # extreme gate weights must not leak
        head.gate_scale.data.fill_(5.0)  # same for the scale
        out_b = head(feats)[0]
    assert torch.equal(out_a[0], out_b[0]), "detection output changed with the gate disabled"
    assert torch.equal(out_a[1], out_b[1]), "prototypes changed with the gate disabled"


def test_gradient_isolation() -> None:
    """gate params get no gradient; trunk/head/aux path keeps gradients."""
    torch.manual_seed(2)
    head = Segment26DSSNoSFCM(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    feats = _feats(2)
    preds = head(feats)
    proto, semantic = preds["proto"]
    (proto.sum() + semantic.sum()).backward()
    for name in ("sem_gate.weight", "sem_gate.bias"):
        p = dict(head.named_parameters())[name]
        assert p.grad is None or torch.all(p.grad == 0), f"{name} must receive no gradient"
    assert head.gate_scale.grad is None or torch.all(head.gate_scale.grad == 0)
    for module_name in ("sem_trunk", "sem_head"):
        params = dict(head.named_parameters())
        grads = [p.grad for n, p in params.items() if n.startswith(module_name)]
        assert all(g is not None for g in grads) and any(torch.any(g != 0) for g in grads), (
            f"{module_name} must stay trained by the auxiliary loss"
        )
    print("[gradients] gate isolated; auxiliary semantic supervision alive")


def test_training_tuple_kept() -> None:
    """Training forward still returns (proto, semantic) for the loss[4] slot."""
    torch.manual_seed(3)
    head = Segment26DSSNoSFCM(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    preds = head(_feats(2))
    assert isinstance(preds["proto"], tuple) and len(preds["proto"]) == 2
    proto, semantic = preds["proto"]
    assert proto.shape == (2, 32, 320, 320) and semantic.shape == (2, 2, 80, 80)


def test_cmp1_checkpoint_compat() -> None:
    """The cmp1 best.pt loads fully into both model variants; only the class flag differs."""
    if not CMP1_CKPT.exists():
        print(f"[compat] SKIP: cmp1 checkpoint not found at {CMP1_CKPT}")
        return
    ckpt = torch.load(CMP1_CKPT, map_location="cpu", weights_only=False)
    csd = ckpt["model"].float().state_dict()
    gated = SegmentationModel(cfg=str(DSS_YAML), ch=3, nc=2, verbose=False)
    ablated = SegmentationModel(cfg=str(NOSFCM_YAML), ch=3, nc=2, verbose=False)
    for name, model in (("gated", gated), ("ablated", ablated)):
        missing, unexpected = model.load_state_dict(csd, strict=False)
        assert not unexpected, f"{name}: unexpected keys {unexpected[:4]}"
        # Only non-persistent num_batches_tracked-style buffers may differ; tensors must all match.
        assert not [k for k in missing if not k.endswith("num_batches_tracked")], (
            f"{name}: missing keys {missing[:4]}"
        )
    # Same weights, different class flag -> outputs must differ once gamma != 0 (cmp1 learned gamma ~2.3 norm).
    gated.eval(), ablated.eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        out_gated = gated(x)[0][1]
        out_abl = ablated(x)[0][1]
    assert not torch.equal(out_gated, out_abl), "cmp1 gamma!=0 must make the two forwards differ"
    print(f"[compat] cmp1 best.pt loads into both models; gate_scale norm={csd['model.23.gate_scale'].norm():.4f}")


def test_official_transfer() -> None:
    """Official ckpt transfer audit is unchanged vs cmp1 (44 new tensors, 14 semseg dropped)."""
    if not OFFICIAL_CKPT.exists():
        print(f"[transfer] SKIP: official checkpoint not found at {OFFICIAL_CKPT}")
        return
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    csd = (ckpt["ema" if ckpt.get("ema") else "model"]).float().state_dict()
    model = SegmentationModel(cfg=str(NOSFCM_YAML), ch=3, nc=80, verbose=False)
    matched = intersect_dicts(csd, model.state_dict())
    new_keys = set(model.state_dict().keys()) - set(matched.keys())
    dropped = set(csd.keys()) - set(matched.keys())
    assert len(new_keys) == 44, f"expected 44 new tensors, got {len(new_keys)}"
    assert all("semseg" in k for k in dropped), f"unexpected dropped tensors: {sorted(dropped)[:4]}"
    print(f"[transfer] matched={len(matched)}/{len(model.state_dict())}, new={len(new_keys)}, semseg dropped={len(dropped)}")


def test_config_matches_baseline() -> None:
    """The nosfcm experiment train block is identical to the stage-0 baseline recipe."""
    import yaml

    with BASELINE_CFG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with NOSFCM_CFG.open("r", encoding="utf-8") as f:
        abl = yaml.safe_load(f)
    assert abl["data"] == base["data"]
    assert abl["train"] == base["train"], "train recipe must match the frozen baseline block"
    assert abl["model"].endswith("yolo26m-dss-nosfcm-seg.yaml")
    print("[config] train block identical to 000 baseline")


def main() -> int:
    tests = [
        test_registration_and_flags,
        test_yaml_parses,
        test_gate_bypassed,
        test_gradient_isolation,
        test_training_tuple_kept,
        test_cmp1_checkpoint_compat,
        test_official_transfer,
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
