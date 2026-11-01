# 水稻害虫实例分割检测系统 V1.0 —— Web 推理 Demo

基于 FastAPI（后端）+ HTML5（前端）的网页版推理演示系统。加载 DSS 最终模型
（`data-v2-cmp1-dss-b16-s42` 的 `best.pt`），在浏览器中上传田间图片即可完成
水稻害虫（稻纵卷叶螟 / 稻螟虫）的实例分割检测。

详细的代码讲解见同目录 [代码讲解.md](代码讲解.md)。

## 目录结构

```text
demo/
├─ app.py              后端：FastAPI 推理服务（模型加载 + 5 个接口）
├─ static/
│  ├─ index.html       前端：页面结构
│  ├─ style.css        前端：样式
│  └─ app.js           前端：交互逻辑
├─ samples/            示例图片（3 张，从验证批次拼图裁切）
└─ 代码讲解.md          代码逐段讲解文档
```

## 启动方式

在项目根目录的 PowerShell 中执行（与仓库其他脚本的运行方式一致）：

```powershell
cd E:\Study\论文撰写\yolo_plus
.\.venv\Scripts\python.exe demo\app.py
```

看到 `请用浏览器打开 http://127.0.0.1:8000` 后，浏览器访问该地址即可。
依赖 `fastapi`、`uvicorn`、`python-multipart`（已装入 `.venv`）。

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--weights` | `runs/data-v2-cmp1-dss-b16-s42/weights/best.pt` | 模型权重路径 |
| `--device` | 自动检测 | 指定 `cpu` 或 `cuda:0` |
| `--host` | `127.0.0.1` | 监听地址（局域网演示可改 `0.0.0.0`） |
| `--port` | `8000` | 端口 |

示例：换一组权重对比效果

```powershell
.\.venv\Scripts\python.exe demo\app.py --weights runs\data-v2-abl-000-y26m-b16-s42\weights\best.pt --port 8001
```

## 功能

- 三种选图方式：点击选择 / 拖拽 / Ctrl+V 粘贴截图；
- 示例图片一键填充；
- 置信度阈值滑块（0.05–0.90），实时生效；
- 检测结果：标注图（分割掩码 + 边界框 + 标签）、实例总数与分类计数、
  分阶段耗时、实例明细表（类别 / 置信度 / 边界框坐标）；
- 原图 / 结果图切换对比，结果图下载；
- 最近 8 次检测历史（内存保存，刷新清空）。

## 更换示例图片为真实数据集图片（推荐）

当前示例图片是从训练拼图（`val_batch*_labels.jpg`）裁切的**降采样图**，
上面残留少量真实标注痕迹，且分辨率只有原图的约 57%，模型检出置信度偏低
（0.15–0.29），仅用于功能演示。

下次云服务器开机时，建议拉取几张原始验证图替换（PowerShell，在项目根目录）：

```powershell
scp autodl:/root/yolo_data/images/val/*.jpg --% 2>$null # 先看文件名
scp autodl:/root/yolo_data/images/val/<某几张文件名> .\demo\samples\
```

文件名包含 `leaffolder` / `stemborer` 时会自动显示对应中文标签（见
`app.py` 的 `SAMPLE_LABELS`）。替换后无需改代码，重启服务即可；
真实分辨率下检测置信度和掩码可视化都会明显更好。

## 常见问题

| 现象 | 处理 |
|---|---|
| 端口被占用 | 加 `--port 8001` 换端口 |
| 控制台中文乱码 | 程序已自动切换 UTF-8；若仍乱码，在 PowerShell 先执行 `$env:PYTHONUTF8='1'` |
| 检出实例为 0 | 调低置信度阈值（如 0.10）；仍为 0 说明图中无目标或图片过小 |
| 想用 CPU | 加 `--device cpu`（推理约 1–2 秒/张） |
| 报"权重文件不存在" | `runs/` 未回传时先从云服务器取回对应 Run，或用 `--weights` 指定已有权重 |

## 与软著材料的关系

`app.py`、`index.html`、`style.css`、`app.js` 四个文件合计约 1380 行，
全部为自研代码，与 `scripts/`、自定义网络模块共同构成软著源程序材料的
"网页检测端"部分；操作手册的"网页检测"一章可直接按本 README 的流程撰写，
截图用浏览器真实运行画面。
