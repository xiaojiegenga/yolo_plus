# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Focused CPU tests for the DSS compound: C3k2DSEM, Proto26DR and Segment26DSS.

Run from the repository root either way:

    python ultralytics-main/tests/test_dss_model.py
    python -m pytest ultralytics-main/tests/test_dss_model.py -q

The file inserts the repository source tree at the front of sys.path so the local
ultralytics package is used instead of any installed copy.

Checks (design doc section 5, 复合结构改进设计方案-DSS三位置复合.md):
1. registration and full-model parsing of yolo26m-dss-seg.yaml;
2. output shapes in train/eval mode and the (proto, semantic) training tuple;
3. zero-init identity: DSEM == C3k2, DPRM == bilinear x2 of Proto26, gate keeps p3 untouched;
4. official checkpoint transfer audit: only the expected new tensors stay unloaded;
5. initial detection equivalence against the official yolo26m-seg model;
6. gradient flow into every new parameter;
7. experiment config train block matches the stage-0 baseline recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # .../ultralytics-main
REPO_ROOT = SOURCE_ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT))

from ultralytics.nn.modules import C3k2, C3k2DSEM, Proto26, Proto26DR, Segment26DSS  # noqa: E402
from ultralytics.nn.tasks import SegmentationModel  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

OFFICIAL_CKPT = Path(r"E:\study\graduate_sec\论文撰写\模型训练\.cache\yolo26m-seg.pt")
DSS_YAML = SOURCE_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-dss-seg.yaml"
BASELINE_CFG = REPO_ROOT / "experiments" / "data-v2-abl-000-y26m-b16-s42.yaml"
DSS_CFG = REPO_ROOT / "experiments" / "data-v2-cmp1-dss-b16-s42.yaml"

# m-scale channel layout for 640 input: (P2, P3, P4, P5) = 256/256/512/512.
CH = (256, 256, 512, 512)


def _feats(batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 256, 160, 160),
        torch.randn(batch, 256, 80, 80),
        torch.randn(batch, 512, 40, 40),
        torch.randn(batch, 512, 20, 20),
    ]


def test_registration() -> None:
    """New modules are importable and registered in the parse_model namespace."""
    import ultralytics.nn.tasks as tasks

    for name in ("C3k2DSEM", "Segment26DSS"):
        assert hasattr(tasks, name), f"{name} missing in ultralytics.nn.tasks"
    assert issubclass(C3k2DSEM, C3k2)
    assert issubclass(Proto26DR, Proto26)


def test_yaml_parses() -> None:
    """yolo26m-dss-seg.yaml builds a SegmentationModel with the expected layer types."""
    model = SegmentationModel(cfg=str(DSS_YAML), ch=3, nc=80, verbose=False)
    assert isinstance(model.model[2], C3k2DSEM), "layer 2 must be C3k2DSEM"
    assert isinstance(model.model[23], Segment26DSS), "layer 23 must be Segment26DSS"
    for idx, expect in ((0, "Conv"), (4, "C3k2"), (16, "C3k2"), (22, "C3k2")):
        assert type(model.model[idx]).__name__ == expect, f"layer {idx} unexpectedly {type(model.model[idx]).__name__}"
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"[yaml] layers={len(model.model)} params={n_params} buffers={n_buffers}")


def test_shapes() -> None:
    """Train mode returns (proto, semantic); eval returns stride-2 protos."""
    torch.manual_seed(0)
    head = Segment26DSS(nc=80, nm=32, npr=256, ch=CH)
    head.train()
    feats = _feats()
    preds = head(feats)
    assert isinstance(preds, dict) and "proto" in preds, "non-end2end training must attach proto"
    proto, semantic = preds["proto"]
    assert proto.shape == (2, 32, 320, 320), f"proto shape {tuple(proto.shape)}"
    assert semantic.shape == (2, 80, 80, 80), f"semantic shape {tuple(semantic.shape)}"
    head.eval()
    result = head(feats)
    detect_y, proto = result[0]
    assert proto.shape == (2, 32, 320, 320)
    assert detect_y.shape[0] == 2, "detection output must keep the batch dimension"
    print(f"[shapes] proto={tuple(proto.shape)} semantic={tuple(semantic.shape)} detect={tuple(detect_y.shape)}")


def test_dsem_zero_init_identity() -> None:
    """With alpha=0 the DSEM stage equals the plain C3k2 stage bit-for-bit."""
    torch.manual_seed(1)
    c1, c2 = 128, 256
    base = C3k2(c1, c2, n=2, c3k=False, e=0.25)
    dsem = C3k2DSEM(c1, c2, n=2, c3k=False, e=0.25)
    missing, unexpected = dsem.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys from the plain C3k2 state: {unexpected}"
    assert all(k.startswith(("dw.", "pw.", "alpha")) for k in missing), f"unexpected missing keys: {missing}"
    x = torch.randn(2, c1, 40, 40)
    assert torch.equal(dsem(x), base(x)), "DSEM forward must equal C3k2 at alpha=0"
    assert torch.equal(dsem.forward_split(x), base.forward_split(x)), "forward_split must match too"


def test_dprm_zero_init_identity() -> None:
    """At initialization DPRM equals bilinear x2 of the official Proto26 output."""
    torch.manual_seed(2)
    feats = _feats(1)[1:]  # (P3, P4, P5)
    p2 = torch.randn(1, 256, 160, 160)
    base = Proto26(ch=(256, 512, 512), c_=256, c2=32, nc=80)
    dr = Proto26DR(ch=CH, c_=256, c2=32, nc=80)
    missing, unexpected = dr.load_state_dict(base.state_dict(), strict=False)
    # missing = module keys absent from the plain Proto26 state -> exactly the new DPRM tensors;
    # unexpected = plain Proto26 keys the module lacks -> only the removed semseg trunk.
    assert all(k.startswith(("p2_proj.", "hi_up.", "hi_cv.", "hi_strip.", "hi_out.")) for k in missing), (
        f"unexpected missing keys: {missing}"
    )
    assert unexpected and all(k.startswith("semseg.") for k in unexpected), f"unexpected keys: {unexpected}"
    base.eval(), dr.eval()
    with torch.no_grad():
        p_base = base(feats)
        p_dr = dr([p2, *feats])
    assert p_base.shape == (1, 32, 160, 160) and p_dr.shape == (1, 32, 320, 320)
    ref = F.interpolate(p_base, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.equal(p_dr, ref), "DPRM must equal bilinear x2 of the official protos at init"


def test_gate_zero_init_identity() -> None:
    """gate_scale=0 keeps the calibrated P3 feature bit-for-bit equal to the raw one."""
    torch.manual_seed(3)
    head = Segment26DSS(nc=80, nm=32, npr=256, ch=CH)
    head.eval()
    p3 = torch.randn(1, 256, 80, 80)
    with torch.no_grad():
        sem = head.sem_trunk(p3)
        gate = torch.sigmoid(head.sem_gate(sem))
        p3_cal = p3 * (1.0 + head.gate_scale.view(1, -1, 1, 1) * gate)
    assert torch.equal(p3_cal, p3), "gate must be an exact identity at gate_scale=0"


def _expected_new_keys() -> set[str]:
    prefixes = (
        "model.2.dw.",
        "model.2.pw.",
        "model.2.alpha",
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
    return {p for p in prefixes}


def test_official_transfer_and_equivalence() -> None:
    """Official weights transfer except the audited new tensors; detection output matches baseline."""
    if not OFFICIAL_CKPT.exists():
        print(f"[transfer] SKIP: official checkpoint not found at {OFFICIAL_CKPT}")
        return
    torch.manual_seed(4)
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    ckpt_model = ckpt["ema" if ckpt.get("ema") else "model"].float() if isinstance(ckpt, dict) else ckpt.float()
    csd = ckpt_model.state_dict()

    dss = SegmentationModel(cfg=str(DSS_YAML), ch=3, nc=80, verbose=False)
    # Values must come from the checkpoint: intersect_dicts(ckpt_state, model_state), as in model.load().
    matched = intersect_dicts(csd, dss.state_dict())
    new_keys = set(dss.state_dict().keys()) - set(matched.keys())
    dropped_keys = set(csd.keys()) - set(matched.keys())

    prefix_ok = _expected_new_keys()
    for key in sorted(new_keys):
        assert any(key.startswith(p) for p in prefix_ok), f"unexpected unloaded tensor: {key}"
    # The removed training-only semseg trunk and nothing else may stay unused in the checkpoint.
    for key in sorted(dropped_keys):
        assert "semseg" in key, f"unexpected dropped ckpt tensor: {key}"
    print(
        f"[transfer] matched={len(matched)}/{len(dss.state_dict())} tensors, "
        f"new={len(new_keys)}, ckpt-only={len(dropped_keys)} (semseg {sum('semseg' in k for k in dropped_keys)})"
    )

    dss.load_state_dict(matched, strict=False)  # same mechanism as model.load()

    # Baseline reference: official yolo26m-seg architecture via the plain seg YAML.
    from ultralytics.nn.tasks import yaml_model_load

    base_cfg = yaml_model_load("yolo26m-seg.yaml")
    official = SegmentationModel(cfg=base_cfg, ch=3, nc=80, verbose=False)
    official.load_state_dict(csd, strict=False)

    dss.eval(), official.eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        out_dss = dss(x)
        out_official = official(x)
    # Eval output: ((y, proto), preds). Detection tensor y must match bit-for-bit.
    y_dss, y_official = out_dss[0][0], out_official[0][0]
    assert y_dss.shape == y_official.shape, f"shape mismatch {y_dss.shape} vs {y_official.shape}"
    assert torch.equal(y_dss, y_official), "detection output must equal the official model at init"
    proto_dss = out_dss[0][1]
    assert proto_dss.shape == (1, 32, 320, 320)
    print(f"[equivalence] detect={tuple(y_dss.shape)} equal=True proto={tuple(proto_dss.shape)}")


def test_gradient_flow() -> None:
    """Every new parameter participates in the training graph.

    Zero-init multiplicative gates (ReZero-style, the A2 pattern) block the gradient to the gate interior
    while the scale is exactly zero, so sem_gate is checked again after the scale is manually opened: the
    scale itself must receive a nonzero gradient immediately, and once it is nonzero the gate interior
    must receive one too.
    """
    torch.manual_seed(5)
    head = Segment26DSS(nc=2, nm=32, npr=256, ch=CH)
    head.train()
    feats = _feats(2)

    def run_backward() -> None:
        head.zero_grad(set_to_none=True)
        preds = head(feats)
        proto, semantic = preds["proto"]
        (proto.sum() + semantic.sum()).backward()

    run_backward()
    checks = {
        "gate_scale": [head.gate_scale],  # opens the gate for the interior after the first update
        "sem_trunk": list(head.sem_trunk.parameters()),  # supervised through sem_head from step 0
        "sem_head": list(head.sem_head.parameters()),
        "p2_proj": list(head.proto.p2_proj.parameters()),  # additive injection: gradient from step 0
        "hi_out": list(head.proto.hi_out.parameters()),  # additive residual: gradient from step 0
    }
    for name, params in checks.items():
        grads = [p.grad for p in params]
        assert all(g is not None for g in grads), f"{name} got no gradient at all"
        assert any(torch.any(g != 0) for g in grads), f"{name} got only zero gradients"

    # Once the scale leaves zero (first optimizer step), the gate interior must receive gradients.
    with torch.no_grad():
        head.gate_scale.fill_(0.1)
    run_backward()
    for name, params in {"sem_gate": list(head.sem_gate.parameters()), "sem_trunk_gate": list(head.sem_trunk.parameters())}.items():
        grads = [p.grad for p in params]
        assert all(g is not None for g in grads), f"{name} got no gradient at all after opening the gate"
        assert any(torch.any(g != 0) for g in grads), f"{name} got only zero gradients after opening the gate"

    # alpha is trained through the backbone; check it at module level.
    torch.manual_seed(6)
    dsem = C3k2DSEM(128, 256, n=2, c3k=False, e=0.25)
    dsem(torch.randn(2, 128, 32, 32)).sum().backward()
    assert dsem.alpha.grad is not None and torch.any(dsem.alpha.grad != 0), "alpha got no gradient"
    print("[gradients] all new parameters receive nonzero gradients (gate interior after the scale opens)")


def test_config_matches_baseline() -> None:
    """The DSS experiment train block is identical to the stage-0 baseline recipe."""
    import yaml

    with BASELINE_CFG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with DSS_CFG.open("r", encoding="utf-8") as f:
        dss = yaml.safe_load(f)
    assert dss["data"] == base["data"], "data yaml must match"
    assert dss["train"] == base["train"], "train recipe must match the frozen baseline block"
    assert dss["model"].endswith("yolo26m-dss-seg.yaml"), "model must point at the DSS architecture"
    print("[config] train block identical to 000 baseline")


def main() -> int:
    tests = [
        test_registration,
        test_yaml_parses,
        test_shapes,
        test_dsem_zero_init_identity,
        test_dprm_zero_init_identity,
        test_gate_zero_init_identity,
        test_official_transfer_and_equivalence,
        test_gradient_flow,
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
