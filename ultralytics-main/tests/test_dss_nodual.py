# Ultralytics AGPL-3.0 License - https://ultralytics.com/legal/license

"""Focused CPU tests for the nodual ablation (Proto26DRNoDual / Segment26DSSNoDual).

Run from the repository root either way:

    python ultralytics-main/tests/test_dss_nodual.py
    python -m pytest ultralytics-main/tests/test_dss_nodual.py -q

Checks:
1. registration + YAML parsing; layer 23 is the ablation head whose prototype module reverts to
   stride 4 while DSEM (layer 2) and the SFCM gate stay enabled;
2. the hi level is truly bypassed: hi weights cannot influence any output; prototypes are 160x160;
3. gradient isolation: hi_* gets no gradient, p2_proj and the gate still learn;
4. the training tuple still carries semantic logits and 160x160 prototypes;
5. at initialization the prototype module equals the baseline Proto26 bit-for-bit (p2_proj zero);
6. the cmp1 checkpoint loads with strict weight parity (state dict keys are identical);
7. official checkpoint transfer audit unchanged (44 new tensors, 14 semseg dropped);
8. experiment config train block matches the stage-0 baseline recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # .../ultralytics-main
REPO_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from ultralytics.nn.modules import (  # noqa: E402
    Proto26,
    Proto26DR,
    Proto26DRNoDual,
    Segment26DSS,
    Segment26DSSNoDual,
)
from ultralytics.nn.tasks import SegmentationModel  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

NODUAL_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-nodual-seg.yaml"
DSS_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-seg.yaml"
OFFICIAL_CKPT = Path(r"E:\study\graduate_sec\论文撰写\模型训练\.cache\yolo26m-seg.pt")
CMP1_CKPT = REPO_ROOT / "runs" / "data-v2-cmp1-dss-b16-s42" / "weights" / "best.pt"
BASELINE_CFG = REPO_ROOT / "experiments" / "data-v2-abl-000-y26m-b16-s42.yaml"
NODUAL_CFG = REPO_ROOT / "experiments" / "data-v2-cmp2-nodual-b16-s42.yaml"

CH = (256, 256, 512, 512)


def _feats(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 256, 160, 160),
        torch.randn(batch, 256, 80, 80),
        torch.randn(batch, 512, 40, 40),
        torch.randn(batch, 512, 20, 20),
    ]


def test_registration_and_flags() -> None:
    """The ablation classes flip only the dual-output flag."""
    import ultralytics.nn.tasks as tasks

    assert hasattr(tasks, "Segment26DSSNoDual"), "Segment26DSSNoDual missing in ultralytics.nn.tasks"
    assert Proto26DR.dual_output is True and Segment26DSS.dual_output is True, "cmp1 defaults must stay on"
    assert Proto26DRNoDual.dual_output is False and Segment26DSSNoDual.dual_output is False
    assert Segment26DSSNoDual.sfcm_gate is True, "the SFCM gate must stay enabled in the nodual run"


def test_yaml_parses() -> None:
    """yolo26m-dss-nodual-seg.yaml builds the ablation head with cmp1's layer layout."""
    model = SegmentationModel(cfg=str(NODUAL_YAML), ch=3, nc=80, verbose=False)
    head = model.model[23]
    assert isinstance(head, Segment26DSSNoDual), "layer 23 must be Segment26DSSNoDual"
    assert type(head.proto).__name__ == "Proto26DRNoDual", "prototype module must be the nodual variant"
    assert head.sfcm_gate is True
    assert type(model.model[2]).__name__ == "C3k2DSEM", "DSEM must stay on layer 2"
    print(f"[yaml] layers={len(model.model)} params={sum(p.numel() for p in model.parameters())}")


def test_hi_level_bypassed() -> None:
    """hi weights cannot influence any forward output; prototypes are 160x160."""
    torch.manual_seed(1)
    head = Segment26DSSNoDual(nc=2, nm=32, npr=256, ch=CH)
    assert torch.count_nonzero(head.proto.p2_proj.conv.weight).item() == 0, "p2_proj must still init at zero"
    head.eval()
    feats = _feats(1)
    with torch.no_grad():
        out_a = head(feats)[0]
        head.proto.hi_out.weight.data.fill_(10.0)  # extreme hi weights must not leak
        head.proto.hi_cv.conv.weight.data.fill_(5.0)
        head.proto.hi_up.weight.data.fill_(3.0)
        out_b = head(feats)[0]
    assert torch.equal(out_a[1], out_b[1]) and out_a[1].shape == (1, 32, 160, 160), (
        "prototypes must stay at stride 4 and ignore hi weights"
    )
    assert torch.equal(out_a[0], out_b[0]), "detection output changed with the hi level disabled"


def test_gradient_isolation() -> None:
    """hi_* gets no gradient; p2_proj and the gate still learn in this variant."""
    torch.manual_seed(2)
    head = Segment26DSSNoDual(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    feats = _feats(2)
    preds = head(feats)
    proto, semantic = preds["proto"]
    (proto.sum() + semantic.sum()).backward()
    params = dict(head.named_parameters())
    for name in params:
        if name.startswith("proto.hi_"):
            assert params[name].grad is None or torch.all(params[name].grad == 0), f"{name} must get no gradient"
    p2_grads = [p.grad for n, p in params.items() if n.startswith("proto.p2_proj")]
    assert all(g is not None for g in p2_grads) and any(torch.any(g != 0) for g in p2_grads), (
        "p2_proj must keep learning (only the hi level is ablated)"
    )
    assert head.gate_scale.grad is not None and torch.any(head.gate_scale.grad != 0), (
        "the SFCM gate must keep learning in the nodual run"
    )
    print("[gradients] hi level isolated; p2_proj and gate still learning")


def test_training_tuple_kept() -> None:
    """Training forward still returns (proto, semantic) with stride-4 prototypes."""
    torch.manual_seed(3)
    head = Segment26DSSNoDual(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    preds = head(_feats(2))
    assert isinstance(preds["proto"], tuple) and len(preds["proto"]) == 2
    proto, semantic = preds["proto"]
    assert proto.shape == (2, 32, 160, 160) and semantic.shape == (2, 2, 80, 80)


def test_initial_equivalence_vs_proto26() -> None:
    """At initialization the nodual prototype module equals the baseline Proto26 bit-for-bit."""
    torch.manual_seed(4)
    feats = _feats(1)[1:]  # (P3, P4, P5)
    p2 = torch.randn(1, 256, 160, 160)
    base = Proto26(ch=(256, 512, 512), c_=256, c2=32, nc=80)
    dr = Proto26DRNoDual(ch=CH, c_=256, c2=32, nc=80)
    missing, unexpected = dr.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith(("p2_proj.", "hi_up.", "hi_cv.", "hi_strip.", "hi_out.")) for k in missing)
    assert unexpected and all(k.startswith("semseg.") for k in unexpected)
    base.eval(), dr.eval()
    with torch.no_grad():
        assert torch.equal(dr([p2, *feats]), base(feats)), "nodual at init must equal the baseline Proto26"


def test_cmp1_checkpoint_compat() -> None:
    """The cmp1 best.pt loads into the nodual model with identical state-dict keys."""
    if not CMP1_CKPT.exists():
        print(f"[compat] SKIP: cmp1 checkpoint not found at {CMP1_CKPT}")
        return
    ckpt = torch.load(CMP1_CKPT, map_location="cpu", weights_only=False)
    csd = ckpt["model"].float().state_dict()
    model = SegmentationModel(cfg=str(NODUAL_YAML), ch=3, nc=2, verbose=False)
    missing, unexpected = model.load_state_dict(csd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected[:4]}"
    assert not [k for k in missing if not k.endswith("num_batches_tracked")], f"missing keys {missing[:4]}"
    model.eval()
    out = model(torch.rand(1, 3, 640, 640))[0]
    assert out[1].shape == (1, 32, 160, 160), "nodual inference must emit stride-4 prototypes"
    print(f"[compat] cmp1 best.pt loads fully; hi_out norm stays at learned {csd['model.23.proto.hi_out.weight'].norm():.4f}")


def test_official_transfer() -> None:
    """Official ckpt transfer audit is unchanged vs cmp1 (44 new tensors, 14 semseg dropped)."""
    if not OFFICIAL_CKPT.exists():
        print(f"[transfer] SKIP: official checkpoint not found at {OFFICIAL_CKPT}")
        return
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    csd = (ckpt["ema" if ckpt.get("ema") else "model"]).float().state_dict()
    model = SegmentationModel(cfg=str(NODUAL_YAML), ch=3, nc=80, verbose=False)
    matched = intersect_dicts(csd, model.state_dict())
    new_keys = set(model.state_dict().keys()) - set(matched.keys())
    dropped = set(csd.keys()) - set(matched.keys())
    assert len(new_keys) == 44, f"expected 44 new tensors, got {len(new_keys)}"
    assert all("semseg" in k for k in dropped), f"unexpected dropped tensors: {sorted(dropped)[:4]}"
    print(f"[transfer] matched={len(matched)}/{len(model.state_dict())}, new={len(new_keys)}, semseg dropped={len(dropped)}")


def test_config_matches_baseline() -> None:
    """The nodual experiment train block is identical to the stage-0 baseline recipe."""
    import yaml

    with BASELINE_CFG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with NODUAL_CFG.open("r", encoding="utf-8") as f:
        abl = yaml.safe_load(f)
    assert abl["data"] == base["data"]
    assert abl["train"] == base["train"], "train recipe must match the frozen baseline block"
    assert abl["model"].endswith("yolo26m-dss-nodual-seg.yaml")
    print("[config] train block identical to 000 baseline")


def main() -> int:
    tests = [
        test_registration_and_flags,
        test_yaml_parses,
        test_hi_level_bypassed,
        test_gradient_isolation,
        test_training_tuple_kept,
        test_initial_equivalence_vs_proto26,
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
