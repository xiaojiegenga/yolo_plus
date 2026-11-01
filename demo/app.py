"""基于深度学习的水稻害虫实例分割检测系统 V1.0 —— Web 推理服务（后端）

本模块基于 FastAPI 框架提供 HTTP 推理服务，是系统的服务端部分，整体结构：

    浏览器（前端页面 demo/static/）
        │  HTTP 请求（上传图片）
        ▼
    app.py（本模块）
        │  调用模型
        ▼
    DSS 实例分割模型（best.pt，定义于 ultralytics-main）
        │  检测结果（掩码 + 边界框 + 置信度）
        ▼
    JSON 响应返回浏览器，前端渲染标注图与统计信息

对外接口一览：
    GET  /                  前端页面 index.html
    GET  /api/info          模型与运行环境信息
    GET  /api/samples       示例图片列表
    GET  /api/samples/{f}   读取一张示例图片
    POST /api/predict       上传图片，返回检测结果

运行方式（在项目根目录执行，详见 demo/README.md）：
    .venv\\Scripts\\python.exe demo/app.py
可选参数：
    --weights  权重文件路径（默认 DSS 最终模型 best.pt）
    --device   cpu / cuda:0（默认自动检测）
    --host / --port  监听地址与端口（默认 127.0.0.1:8000）
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# 运行环境准备。注意：这些代码必须在导入 ultralytics 之前执行。
# --------------------------------------------------------------------------

# 本文件位于 demo/ 目录下，项目根目录是它的上一级
REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
STATIC_DIR = DEMO_DIR / "static"
SAMPLES_DIR = DEMO_DIR / "samples"

# Ultralytics 的用户级配置目录统一放在项目内的 .cache 下，
# 与仓库其他脚本（见 LOCAL_SETUP.local.md）的约定一致，避免污染用户主目录
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))

# Windows 控制台默认 GBK 编码，直接打印中文可能乱码或报错，切换为 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # 某些环境下不存在 reconfigure 方法，忽略即可

# 自定义网络结构（Segment26DSS、Proto26DR 等）定义在仓库内的 ultralytics-main，
# 必须把它加入模块搜索路径，加载权重时才能反序列化出这些自定义类
sys.path.insert(0, str(REPO_ROOT / "ultralytics-main"))

import numpy as np  # noqa: E402  (以上是环境准备，导入语句因此放在这里)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402

from ultralytics import YOLO  # noqa: E402

# --------------------------------------------------------------------------
# 常量配置
# --------------------------------------------------------------------------

SOFTWARE_NAME = "基于深度学习的水稻害虫实例分割检测系统"
SOFTWARE_VERSION = "V1.0"

# 默认使用 DSS 最终模型的正式权重（Run: data-v2-cmp1-dss-b16-s42，已冻结）
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "data-v2-cmp1-dss-b16-s42" / "weights" / "best.pt"

# 推理输入尺寸 640，与正式实验协议（imgsz 640）保持一致
INPUT_SIZE = 640

# 类别的中文显示名与前端配色。key 必须与训练数据集的类别名完全一致；
# 遇到未知类别时回退为原名 + 默认紫色，保证程序不会因类别表对不上而崩溃
CLASS_META = {
    "Rice leaffolder": {"name_zh": "稻纵卷叶螟", "color": "#f59e0b"},
    "Rice stemborers": {"name_zh": "稻螟虫", "color": "#06b6d4"},
}
FALLBACK_COLOR = "#8b5cf6"

# 上传限制：20MB 覆盖常见手机照片；格式限定为常见图片类型
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 示例图片文件名 -> 展示标签 的映射，用于 /api/samples 接口
SAMPLE_LABELS = {
    "leaffolder": "田间样本 · 稻纵卷叶螟",
    "stemborer": "田间样本 · 稻螟虫",
}


def sample_label(filename: str) -> str:
    """根据示例图片的文件名生成展示标签。"""
    for key, label in SAMPLE_LABELS.items():
        if key in filename.lower():
            return label
    return "田间样本"


# --------------------------------------------------------------------------
# 应用与模型状态
# --------------------------------------------------------------------------

app = FastAPI(title=f"{SOFTWARE_NAME} {SOFTWARE_VERSION} 推理服务")

# 模型相关状态。init_model() 在服务启动前调用，把这几个全局变量填充好。
# 单进程单模型场景下用模块级全局变量保存状态最直接，也便于阅读。
MODEL: YOLO | None = None
DEVICE: str = "cpu"
DEVICE_NAME: str = "cpu"
WEIGHTS_USED: Path = DEFAULT_WEIGHTS


def init_model(weights: Path, device: str | None = None) -> None:
    """加载模型权重并完成一次预热推理。

    预热的原因：第一次推理包含显存分配、算子编译等一次性开销，
    提前用一张空白图跑一遍，用户第一次真实检测的耗时就具有可比性。
    """
    global MODEL, DEVICE, DEVICE_NAME, WEIGHTS_USED

    if not weights.exists():
        raise FileNotFoundError(
            f"权重文件不存在：{weights}\n"
            f"请确认 runs/ 目录已从云服务器回传，或用 --weights 指定其他路径"
        )

    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = device

    if device.startswith("cuda"):
        import torch

        try:
            DEVICE_NAME = f"{device} ({torch.cuda.get_device_name(0)})"
        except Exception:
            DEVICE_NAME = device
    else:
        DEVICE_NAME = device

    WEIGHTS_USED = weights
    print(f"[demo] 软件名称：{SOFTWARE_NAME} {SOFTWARE_VERSION}")
    print(f"[demo] 加载权重：{weights}")
    print(f"[demo] 推理设备：{DEVICE_NAME}")

    MODEL = YOLO(str(weights))

    # 预热推理：输入一张全黑图即可，目的只是触发初始化
    MODEL.predict(
        source=np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8),
        imgsz=INPUT_SIZE,
        device=DEVICE,
        verbose=False,
    )
    print(f"[demo] 模型就绪，类别表：{MODEL.names}")


def count_parameters(model: YOLO) -> int:
    """统计模型参数量（用于 /api/info 展示）。"""
    try:
        return sum(p.numel() for p in model.model.parameters())
    except Exception:
        return 0


# --------------------------------------------------------------------------
# 页面与静态资源
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    """返回前端主页。"""
    return FileResponse(STATIC_DIR / "index.html")


# 托管 CSS / JS 等静态文件，浏览器通过 /static/xxx.css 访问
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------
# API 接口
# --------------------------------------------------------------------------


@app.get("/api/info")
def get_info() -> dict:
    """模型与运行环境信息，前端页头的能力标签由此生成。"""
    classes = []
    for class_id, class_name in MODEL.names.items():
        meta = CLASS_META.get(class_name, {})
        classes.append(
            {
                "id": class_id,
                "name": class_name,
                "name_zh": meta.get("name_zh", class_name),
                "color": meta.get("color", FALLBACK_COLOR),
            }
        )
    # run 目录名：正常路径形如 runs/<run-id>/weights/best.pt，取祖父目录；
    # 用户直接指定散落文件时退化为父目录名，保证字段始终有合理取值
    parent_name = WEIGHTS_USED.parent.name
    run_id = (
        WEIGHTS_USED.parent.parent.name if parent_name.lower() == "weights" else parent_name
    )
    return {
        "software": SOFTWARE_NAME,
        "version": SOFTWARE_VERSION,
        "weights": WEIGHTS_USED.name,
        "run_id": run_id,
        "device": DEVICE_NAME,
        "input_size": INPUT_SIZE,
        "parameters": count_parameters(MODEL),
        "classes": classes,
    }


@app.get("/api/samples")
def list_samples() -> dict:
    """列出 demo/samples 目录下的示例图片。"""
    items = [
        {"file": path.name, "label": sample_label(path.name)}
        for path in sorted(SAMPLES_DIR.glob("*.jpg"))
    ]
    return {"items": items}


@app.get("/api/samples/{name}")
def get_sample(name: str) -> FileResponse:
    """读取一张示例图片。

    安全校验：只允许访问 samples 目录内的文件。如果直接拼接用户传入的
    name 而不检查，攻击者可以用 ../ 访问服务器上的任意文件（路径穿越）。
    """
    path = (SAMPLES_DIR / name).resolve()
    if path.parent != SAMPLES_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="示例图片不存在")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(..., description="待检测图片"),
    conf: float = Form(0.25, description="置信度阈值"),
) -> dict:
    """核心接口：接收一张图片，返回标注结果图与实例列表。

    前端用 FormData（multipart/form-data）提交；响应为 JSON，
    其中标注图以 base64 编码的 JPEG 字符串返回，前端直接放进 <img> 的 src。

    说明：本接口是 async 的（为了异步读取上传内容），推理本身是同步阻塞的，
    对单人使用的演示系统没有问题；如需高并发应把推理放入线程池。
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # ---- 1. 校验文件类型与大小 ------------------------------------------
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式 {suffix or '(未知)'}，请上传 jpg/png/bmp/webp",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="图片超过 20MB，请压缩后重试")

    # ---- 2. 字节流 -> PIL 图像 -> numpy 数组 -----------------------------
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="图片解析失败，文件可能已损坏")

    # Ultralytics 约定 numpy 输入使用 OpenCV 的 BGR 通道顺序，
    # 而 PIL 打开是 RGB，因此交换通道（[:, :, ::-1] 表示反转第三维）
    bgr_array = np.asarray(image)[:, :, ::-1]

    # ---- 3. 推理 ---------------------------------------------------------
    start = time.perf_counter()
    results = MODEL.predict(
        source=bgr_array,
        imgsz=INPUT_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    result = results[0]  # 单张图输入，结果列表里只有一个元素

    # ---- 4. 生成标注图并编码为 base64 ------------------------------------
    # plot() 会在原图上绘制半透明分割掩码、边界框和类别标签，返回 BGR 数组
    annotated_bgr = result.plot(line_width=2, font_size=14)
    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB，供 PIL 使用
    buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buffer, format="JPEG", quality=88)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    # ---- 5. 提取实例信息（类别、置信度、边界框坐标） ----------------------
    instances = []
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            meta = CLASS_META.get(class_name, {})
            instances.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_name_zh": meta.get("name_zh", class_name),
                    "color": meta.get("color", FALLBACK_COLOR),
                    "confidence": round(float(result.boxes.conf[i]), 4),
                    # xyxy 边界框：[左上 x, 左上 y, 右下 x, 右下 y]，像素坐标
                    "bbox_xyxy": [round(float(v), 1) for v in result.boxes.xyxy[i].tolist()],
                }
            )
    # 按置信度从高到低排序，前端表格直接按此顺序渲染
    instances.sort(key=lambda item: item["confidence"], reverse=True)

    # result.speed 是 Ultralytics 记录的分阶段耗时（毫秒）：
    # preprocess（预处理）/ inference（网络前向）/ postprocess（后处理）
    speed = {key: round(float(value), 2) for key, value in result.speed.items()}

    return {
        "software": f"{SOFTWARE_NAME} {SOFTWARE_VERSION}",
        "conf_threshold": conf,
        "device": DEVICE_NAME,
        "image": image_base64,
        "instance_count": len(instances),
        "instances": instances,
        "speed": speed,
        "wall_ms": round(wall_ms, 2),
        "source_size": {"width": image.width, "height": image.height},
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    """兜底错误处理：未预期的异常统一返回 JSON，避免前端收到 HTML 错误页。"""
    return JSONResponse(status_code=500, content={"detail": f"服务内部错误：{exc}"})


# --------------------------------------------------------------------------
# 启动入口
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{SOFTWARE_NAME} Web 推理服务")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    parser.add_argument("--device", default=None, help="cpu / cuda:0，默认自动检测")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    init_model(args.weights, args.device)

    import uvicorn

    print(f"[demo] 服务启动中，请用浏览器打开 http://{args.host}:{args.port}")
    # log_level="warning"：屏蔽 uvicorn 每个请求的访问日志，控制台保持清爽
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
