# Ultralytics AGPL-3.0 License - https://www.ultralytics.com/legal/license

"""Focused CPU tests for the nop2inj ablation (Proto26DRNoP2 / Segment26DSSNoP2).

Run from the repository root either way:

    python ultralytics-main/tests/test_dss_nop2inj.py
    python -m pytest ultralytics-main/tests/test_dss_nop2inj.py -q

Checks:
1. registration + YAML parsing; layer 23 is the ablation head whose prototype module ignores the
   P2 input while DSEM (layer 2), the SFCM gate and the dual-resolution branch stay enabled;
2. the injection is truly bypassed: neither p2_proj weights nor the P2 input can influence any
   output; prototypes stay at stride 2;
3. gradient isolation: p2_proj gets no gradient, the gate / hi level / semantic branch still learn;
4. the training tuple still carries semantic logits and stride-2 prototypes;
5. at initialization the module equals the full Proto26DR and the baseline Proto26 up to
   bilinear x2 (p2_proj is zero in both anyway);
6. the cmp1 checkpoint loads with identical state-dict keys;
7. official checkpoint transfer audit unchanged (44 new tensors, 14 semseg dropped);
8. experiment config train block matches the stage-0 baseline recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # .../ultralytics-main
REPO_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from ultralytics.nn.modules import Proto26, Proto26DR, Proto26DRNoP2, Segment26DSSNoP2  # noqa: E402
from ultralytics.nn.tasks import SegmentationModel  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

NOP2INJ_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-nop2inj-seg.yaml"
OFFICIAL_CKPT = Path(r"E:\study\graduate_sec\论文撰写\模型训练\.cache\yolo26m-seg.pt")
CMP1_CKPT = REPO_ROOT / "runs" / "data-v2-cmp1-dss-b16-s42" / "weights" / "best.pt"
BASELINE_CFG = REPO_ROOT / "experiments" / "data-v2-abl-000-y26m-b16-s42.yaml"
NOP2INJ_CFG = REPO_ROOT / "experiments" / "data-v2-cmp2-nop2inj-b16-s42.yaml"

CH = (256, 256, 512, 512)


def _feats(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 256, 160, 160),
        torch.randn(batch, 256, 80, 80),
        torch.randn(batch, 512, 40, 40),
        torch.randn(batch, 512, 20, 20),
    ]


def test_registration_and_flags() -> None:
    """The ablation classes flip only the p2_inject flag."""
    import ultralytics.nn.tasks as tasks

    assert hasattr(tasks, "Segment26DSSNoP2"), "Segment26DSSNoP2 missing in ultralytics.nn.tasks"
    assert Proto26DR.p2_inject is True, "cmp1 default must keep the injection on"
    assert Proto26DRNoP2.p2_inject is False and Segment26DSSNoP2.p2_inject is False
    assert Segment26DSSNoP2.sfcm_gate is True and Segment26DSSNoP2.dual_output is True, (
        "gate and dual-resolution must stay enabled in the nop2inj run"
    )


def test_yaml_parses() -> None:
    """yolo26m-dss-nop2inj-seg.yaml builds the ablation head with cmp1's layer layout."""
    model = SegmentationModel(cfg=str(NOP2INJ_YAML), ch=3, nc=80, verbose=False)
    head = model.model[23]
    assert isinstance(head, Segment26DSSNoP2), "layer 23 must be Segment26DSSNoP2"
    assert type(head.proto).__name__ == "Proto26DRNoP2", "prototype module must be the nop2inj variant"
    assert type(model.model[2]).__name__ == "C3k2DSEM", "DSEM must stay on layer 2"
    print(f"[yaml] layers={len(model.model)} params={sum(p.numel() for p in model.parameters())}")


def test_injection_bypassed() -> None:
    """Neither p2_proj weights nor the P2 input can influence any forward output."""
    torch.manual_seed(1)
    head = Segment26DSSNoP2(nc=2, nm=32, npr=256, ch=CH)
    assert torch.count_nonzero(head.proto.hi_out.weight).item() == 0, "hi_out must still init at zero"
    head.eval()
    feats = _feats(1)
    with torch.no_grad():
        out_a = head(feats)[0]
        head.proto.p2_proj.conv.weight.data.fill_(10.0)  # extreme injection weights must not leak
        feats_b = [feats[0] * 50.0, *feats[1:]]  # same P3/P4/P5, wildly different P2 input
        out_b = head(feats_b)[0]
    assert torch.equal(out_a[0], out_b[0]) and torch.equal(out_a[1], out_b[1]), (
        "outputs must ignore both p2_proj weights and the P2 input"
    )
    assert out_a[1].shape == (1, 32, 320, 320), "prototypes must stay at stride 2"


def test_gradient_isolation() -> None:
    """p2_proj gets no gradient; gate, hi level and semantic branch still learn."""
    torch.manual_seed(2)
    head = Segment26DSSNoP2(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    preds = head(_feats(2))
    proto, semantic = preds["proto"]
    (proto.sum() + semantic.sum()).backward()
    params = dict(head.named_parameters())
    p2_grads = [p.grad for n, p in params.items() if n.startswith("proto.p2_proj")]
    assert all(g is None or torch.all(g == 0) for g in p2_grads), "p2_proj must get no gradient"
    for group in ("proto.hi_", "sem_trunk", "sem_head"):
        grads = [p.grad for n, p in params.items() if n.startswith(group)]
        assert all(g is not None for g in grads) and any(torch.any(g != 0) for g in grads), f"{group} must learn"
    assert head.gate_scale.grad is not None and torch.any(head.gate_scale.grad != 0), (
        "the SFCM gate must keep learning in the nop2inj run"
    )
    print("[gradients] p2_proj isolated; gate / hi level / semantic branch still learning")


def test_training_tuple_kept() -> None:
    """Training forward still returns (proto, semantic) with stride-2 prototypes."""
    torch.manual_seed(3)
    head = Segment26DSSNoP2(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    preds = head(_feats(2))
    assert isinstance(preds["proto"], tuple) and len(preds["proto"]) == 2
    proto, semantic = preds["proto"]
    assert proto.shape == (2, 32, 320, 320) and semantic.shape == (2, 2, 80, 80)


def test_initial_equivalence() -> None:
    """At init the nop2inj module equals the full Proto26DR and bilinear x2 of the baseline."""
    torch.manual_seed(4)
    feats = _feats(1)[1:]  # (P3, P4, P5)
    p2 = torch.randn(1, 256, 160, 160)
    base = Proto26(ch=(256, 512, 512), c_=256, c2=32, nc=80)
    full = Proto26DR(ch=CH, c_=256, c2=32, nc=80)
    ablated = Proto26DRNoP2(ch=CH, c_=256, c2=32, nc=80)
    for module in (full, ablated):
        missing, unexpected = module.load_state_dict(base.state_dict(), strict=False)
        assert all(k.startswith(("p2_proj.", "hi_up.", "hi_cv.", "hi_strip.", "hi_out.")) for k in missing)
        assert unexpected and all(k.startswith("semseg.") for k in unexpected)
    base.eval(), full.eval(), ablated.eval()
    with torch.no_grad():
        ref = F.interpolate(base(feats), scale_factor=2, mode="bilinear", align_corners=False)
        assert torch.equal(ablated([p2, *feats]), ref), "nop2inj at init must equal bilinear x2 of Proto26"
        assert torch.equal(ablated([p2, *feats]), full([p2, *feats])), "init equals the full Proto26DR"


def test_cmp1_checkpoint_compat() -> None:
    """The cmp1 best.pt loads into the nop2inj model with identical state-dict keys."""
    if not CMP1_CKPT.exists():
        print(f"[compat] SKIP: cmp1 checkpoint not found at {CMP1_CKPT}")
        return
    ckpt = torch.load(CMP1_CKPT, map_location="cpu", weights_only=False)
    csd = ckpt["model"].float().state_dict()
    model = SegmentationModel(cfg=str(NOP2INJ_YAML), ch=3, nc=2, verbose=False)
    missing, unexpected = model.load_state_dict(csd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected[:4]}"
    assert not [k for k in missing if not k.endswith("num_batches_tracked")], f"missing keys {missing[:4]}"
    model.eval()
    out = model(torch.rand(1, 3, 640, 640))[0]
    assert out[1].shape == (1, 32, 320, 320), "nop2inj inference must emit stride-2 prototypes"
    print(f"[compat] cmp1 best.pt loads fully; p2_proj norm stays at learned {csd['model.23.proto.p2_proj.conv.weight'].norm():.4f}")


def test_official_transfer() -> None:
    """Official ckpt transfer audit is unchanged vs cmp1 (44 new tensors, 14 semseg dropped)."""
    if not OFFICIAL_CKPT.exists():
        print(f"[transfer] SKIP: official checkpoint not found at {OFFICIAL_CKPT}")
        return
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    csd = (ckpt["ema" if ckpt.get("ema") else "model"]).float().state_dict()
    model = SegmentationModel(cfg=str(NOP2INJ_YAML), ch=3, nc=80, verbose=False)
    matched = intersect_dicts(csd, model.state_dict())
    new_keys = set(model.state_dict().keys()) - set(matched.keys())
    dropped = set(csd.keys()) - set(matched.keys())
    assert len(new_keys) == 44, f"expected 44 new tensors, got {len(new_keys)}"
    assert all("semseg" in k for k in dropped), f"unexpected dropped tensors: {sorted(dropped)[:4]}"
    print(f"[transfer] matched={len(matched)}/{len(model.state_dict())}, new={len(new_keys)}, semseg dropped={len(dropped)}")


def test_config_matches_baseline() -> None:
    """The nop2inj experiment train block is identical to the stage-0 baseline recipe."""
    import yaml

    with BASELINE_CFG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with NOP2INJ_CFG.open("r", encoding="utf-8") as f:
        abl = yaml.safe_load(f)
    assert abl["data"] == base["data"]
    assert abl["train"] == base["train"], "train recipe must match the frozen baseline block"
    assert abl["model"].endswith("yolo26m-dss-nop2inj-seg.yaml")
    print("[config] train block identical to 000 baseline")


def main() -> int:
    tests = [
        test_registration_and_flags,
        test_yaml_parses,
        test_injection_bypassed,
        test_gradient_isolation,
        test_training_tuple_kept,
        test_initial_equivalence,
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
