/* ==========================================================================
   水稻害虫实例分割检测系统 V1.0 —— 前端交互逻辑
   --------------------------------------------------------------------------
   浏览器端的全部行为集中在这个文件：

     1. 页面加载时向 /api/info、/api/samples 要基础信息并渲染；
     2. 三种选图方式（点击 / 拖拽 / Ctrl+V 粘贴）统一走 setFile()；
     3. 点击「开始检测」后，用 fetch 把图片以 FormData 形式 POST 到
        /api/predict，拿到 JSON 结果再渲染标注图、统计卡片、明细表；
     4. 历史记录只保存在内存里（刷新页面即清空），足够演示使用。
   ========================================================================== */

"use strict";

/* ------------------------------ 全局状态 ------------------------------ */

/**
 * state 对象集中保存页面当前的数据：
 * - file       用户当前选中的图片（File 对象，来自选择框/拖拽/粘贴）
 * - originalUrl 原图的本地预览地址（URL.createObjectURL 生成）
 * - result     最近一次 /api/predict 的完整响应
 * - history    历史检测记录数组（最多保留 8 条）
 */
const state = {
  file: null,
  originalUrl: null,
  result: null,
  history: [],
};

/** 快捷获取 DOM 元素：el("detect-btn") 等价于 document.getElementById(...) */
const el = (id) => document.getElementById(id);

/* ------------------------------ 工具函数 ------------------------------ */

/** 在页面底部弹出短暂提示（3 秒自动消失），用于错误与操作反馈。 */
function toast(message) {
  const node = el("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast._timer); // 连续调用时重置计时，避免提示提前消失
  toast._timer = setTimeout(() => (node.hidden = true), 3000);
}

/** 更新左栏状态文字；isError 为真时显示为红色错误样式。 */
function setStatus(message, isError = false) {
  const node = el("status");
  node.textContent = message;
  node.classList.toggle("error", isError);
}

/** 数字格式化：保留 decimals 位小数并去掉多余的 0（0.7500 -> 0.75）。 */
function fmtNumber(value, decimals = 2) {
  return Number(value).toFixed(decimals).replace(/\.?0+$/, "");
}

/** 请求出错时，从 Response 对象中尽量解析出后端给的中文错误信息。 */
async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || `请求失败（HTTP ${response.status}）`;
  } catch {
    return `请求失败（HTTP ${response.status}）`;
  }
}

/* ------------------------------ 页面初始化 ------------------------------ */

/** 页面加载完成后最先执行：拉取模型信息与示例图片列表。 */
async function init() {
  await loadInfo();
  await loadSamples();
  bindEvents();
}

/** 请求 /api/info，把设备、权重、类别等信息渲染成页头标签。 */
async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) throw new Error(await readError(response));
    const info = await response.json();

    state.info = info; // 类别色等信息在渲染表格时还要用到

    const paramsM = info.parameters ? `${fmtNumber(info.parameters / 1e6, 1)}M` : "--";
    const chips = [
      `设备 ${info.device}`,
      `权重 ${info.run_id} / best.pt`,
      `参数量 ${paramsM}`,
      `输入 ${info.input_size}×${info.input_size}`,
      ...info.classes.map((c) => c.name_zh),
    ];
    el("meta-chips").innerHTML = chips
      .map((text) => `<span class="chip">${text}</span>`)
      .join("");
  } catch (error) {
    toast(`模型信息加载失败：${error.message}`);
  }
}

/** 请求 /api/samples，为每张示例图生成一个缩略图按钮。 */
async function loadSamples() {
  const container = el("samples");
  try {
    const response = await fetch("/api/samples");
    if (!response.ok) throw new Error(await readError(response));
    const { items } = await response.json();

    container.innerHTML = "";
    if (items.length === 0) {
      container.innerHTML = `<p class="hint">暂无示例图片</p>`;
      return;
    }
    for (const item of items) {
      // 每个按钮 = 缩略图 + 标签；点击后把示例图当作用户上传的图片处理
      const button = document.createElement("button");
      button.className = "sample-btn";
      button.title = item.label;
      button.innerHTML =
        `<img src="/api/samples/${encodeURIComponent(item.file)}" alt="${item.label}">` +
        `<span>${item.label}</span>`;
      button.addEventListener("click", async () => {
        try {
          const imageResponse = await fetch(`/api/samples/${encodeURIComponent(item.file)}`);
          if (!imageResponse.ok) throw new Error(await readError(imageResponse));
          const blob = await imageResponse.blob();
          setFile(new File([blob], item.file, { type: blob.type }));
        } catch (error) {
          toast(`示例图片加载失败：${error.message}`);
        }
      });
      container.appendChild(button);
    }
  } catch (error) {
    container.innerHTML = `<p class="hint">示例图片加载失败：${error.message}</p>`;
  }
}

/* ------------------------------ 选图与预览 ------------------------------ */

/** 校验并"选中"一张图片：保存到 state、生成预览、启用检测按钮。 */
function setFile(file) {
  if (!file) return;

  if (!/^image\/(jpeg|png|bmp|webp)$/.test(file.type)) {
    toast("不支持的图片格式，请选择 jpg / png / bmp / webp");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    toast("图片超过 20MB，请压缩后重试");
    return;
  }

  // 释放上一张图的预览地址，避免内存泄漏
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);

  state.file = file;
  state.result = null;
  state.originalUrl = URL.createObjectURL(file); // 生成本地临时 URL 用于预览

  // 显示原图预览（结果出来前先让用户看到自己选了什么）
  const viewerImg = el("viewer-img");
  viewerImg.src = state.originalUrl;
  viewerImg.hidden = false;
  el("placeholder").hidden = true;
  el("show-orig").checked = true; // 预览阶段默认显示原图
  el("download-btn").hidden = true;

  el("detect-btn").disabled = false;
  el("stats").innerHTML = "";
  el("table-wrap").hidden = true;
  setStatus(`已选择 ${file.name || "剪贴板图片"}（${fmtNumber(file.size / 1024, 0)} KB）`);
}

/* ------------------------------ 检测主流程 ------------------------------ */

/** 点击「开始检测」：上传图片 -> 等待推理 -> 渲染结果。 */
async function detect() {
  if (!state.file) return;

  const button = el("detect-btn");
  button.disabled = true;
  button.classList.add("busy");
  button.textContent = "检测中……";
  setStatus("正在上传图片并推理，请稍候");

  try {
    // FormData 对应 HTTP 的 multipart/form-data，
    // 与后端 predict(file: UploadFile, conf: Form) 的参数名一一对应
    const formData = new FormData();
    formData.append("file", state.file);
    formData.append("conf", el("conf").value);

    const response = await fetch("/api/predict", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await readError(response));

    const data = await response.json();
    state.result = data;
    renderResult(data);
    appendHistory(data);
    setStatus(
      `检测完成：${data.instance_count} 个实例 · 端到端 ${fmtNumber(data.wall_ms, 0)} ms`
    );
  } catch (error) {
    setStatus(error.message, true);
    toast(error.message);
  } finally {
    // 无论成功失败都恢复按钮状态
    button.disabled = false;
    button.classList.remove("busy");
    button.textContent = "开始检测";
  }
}

/* ------------------------------ 结果渲染 ------------------------------ */

/** 把 /api/predict 的 JSON 响应渲染到右栏：标注图 + 统计卡片 + 明细表。 */
function renderResult(data) {
  // 标注图：后端返回 base64 JPEG，加上 data: 前缀即可作为 <img> 的 src
  const viewerImg = el("viewer-img");
  viewerImg.src = `data:image/jpeg;base64,${data.image}`;
  viewerImg.hidden = false;
  el("placeholder").hidden = true;
  el("show-orig").checked = false; // 默认展示检测结果
  el("download-btn").hidden = false;

  // ---- 统计卡片 ----
  const perClass = {};
  for (const instance of data.instances) {
    perClass[instance.class_name_zh] = (perClass[instance.class_name_zh] || 0) + 1;
  }
  const cards = [
    `<div class="stat-card"><div class="num">${data.instance_count}</div><div class="label">检出实例总数</div></div>`,
  ];
  for (const [name, count] of Object.entries(perClass)) {
    const color = data.instances.find((i) => i.class_name_zh === name).color;
    cards.push(
      `<div class="stat-card tinted" style="--accent:${color}">` +
        `<div class="num">${count}</div><div class="label">${name}</div></div>`
    );
  }
  cards.push(
    `<div class="stat-card"><div class="num">${fmtNumber(data.speed.inference, 1)}</div><div class="label">网络前向 (ms)</div></div>`,
    `<div class="stat-card"><div class="num">${fmtNumber(data.wall_ms, 0)}</div><div class="label">端到端耗时 (ms)</div></div>`
  );
  el("stats").innerHTML = cards.join("");

  // ---- 实例明细表 ----
  const tbody = el("inst-tbody");
  if (data.instance_count === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color:#6b7280">当前阈值下未检出目标，可尝试调低置信度阈值</td></tr>`;
  } else {
    tbody.innerHTML = data.instances
      .map(
        (instance, index) =>
          `<tr>` +
          `<td>${index + 1}</td>` +
          `<td><span class="class-dot" style="background:${instance.color}"></span>${instance.class_name_zh}</td>` +
          `<td>${instance.confidence.toFixed(4)}</td>` +
          `<td>${instance.bbox_xyxy.map((v) => Math.round(v)).join(", ")}</td>` +
          `</tr>`
      )
      .join("");
  }
  el("table-wrap").hidden = false;
}

/* ------------------------------ 历史记录 ------------------------------ */

/** 把一次检测结果追加到历史区（仅保存在内存中，最多 8 条）。 */
function appendHistory(data) {
  state.history.unshift(data);
  if (state.history.length > 8) state.history.pop();

  const container = el("history");
  container.innerHTML = "";
  state.history.forEach((record) => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML =
      `<img src="data:image/jpeg;base64,${record.image}" alt="历史结果">` +
      `<div class="hi-meta">${record.instance_count} 实例 · ${fmtNumber(record.wall_ms, 0)}ms</div>`;
    // 点击历史条目可回看该次结果
    item.addEventListener("click", () => {
      state.result = record;
      renderResult(record);
      el("conf").value = record.conf_threshold;
      el("conf-val").textContent = Number(record.conf_threshold).toFixed(2);
    });
    container.appendChild(item);
  });
}

/* ------------------------------ 事件绑定 ------------------------------ */

/** 集中绑定所有 UI 事件，init() 的最后一步。 */
function bindEvents() {
  // 1. 点击 dropzone -> 触发隐藏的文件选择框
  const dropzone = el("dropzone");
  const fileInput = el("file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) setFile(fileInput.files[0]);
    fileInput.value = ""; // 清空选择，否则连续选同一张图不触发 change
  });

  // 2. 拖拽：dragover 时高亮边框，drop 时取第一个文件
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault(); // 必须阻止默认行为，否则浏览器直接打开图片
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    if (file) setFile(file);
  });

  // 3. Ctrl+V 粘贴截图：监听整个文档的 paste 事件
  document.addEventListener("paste", (event) => {
    const item = Array.from(event.clipboardData.items).find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  // 4. 置信度滑块：实时回显数值
  const confSlider = el("conf");
  confSlider.addEventListener("input", () => (el("conf-val").textContent = Number(confSlider.value).toFixed(2)));

  // 5. 检测按钮
  el("detect-btn").addEventListener("click", detect);

  // 6. 原图 / 结果图切换
  el("show-orig").addEventListener("change", (event) => {
    if (!state.originalUrl) return;
    el("viewer-img").src = event.target.checked
      ? state.originalUrl
      : state.result
        ? `data:image/jpeg;base64,${state.result.image}`
        : state.originalUrl;
  });

  // 7. 下载结果图：构造一个临时 <a> 触发浏览器下载
  el("download-btn").addEventListener("click", () => {
    if (!state.result) return;
    const link = document.createElement("a");
    link.href = `data:image/jpeg;base64,${state.result.image}`;
    link.download = `dss-result-${Date.now()}.jpg`;
    link.click();
  });
}

// DOM 就绪后启动（script 标签在 body 末尾，此时元素都已存在）
init();
