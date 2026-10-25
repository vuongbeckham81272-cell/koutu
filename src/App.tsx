import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent as ReactDragEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { downloadDir } from "@tauri-apps/api/path";
import PreviewView from "./components/PreviewView";
import BatchList from "./components/BatchList";
import type { BatchItem, ErasePoint, ModelStatus, Stage } from "./types";
import "./App.css";

const SUPPORTED = ["png", "jpg", "jpeg", "webp", "bmp"];

function extOf(path: string) {
  return path.split(".").pop()?.toLowerCase() ?? "";
}

function baseName(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

// 抠图结果永远是透明 PNG（修复：此前保留原扩展名导致导出 .jpg 变白底）
function cutoutName(name: string, suffix = "-cutout") {
  const dot = name.lastIndexOf(".");
  const base = dot > 0 ? name.slice(0, dot) : name;
  return `${base}${suffix}.png`;
}

function bytesToDataUrl(bytes: Uint8Array, mime = "image/png") {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return `data:${mime};base64,${btoa(bin)}`;
}

function uid() {
  return (
    crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`
  );
}

function App() {
  const [dragOver, setDragOver] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [modelError, setModelError] = useState("");
  const [backend, setBackend] = useState<"gpu" | "cpu">("cpu");
  const [gpuAccel, setGpuAccel] = useState(true);
  const [stage, setStage] = useState<Stage>("idle");
  const [status, setStatus] = useState("");
  const [hardEdge, setHardEdge] = useState(false);
  const [smoothEdge, setSmoothEdge] = useState(true);
  const [view, setView] = useState<"original" | "result">("result");
  const [eraseMode, setEraseMode] = useState(false);
  const [tolerance, setTolerance] = useState(30);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultBytes, setResultBytes] = useState<Uint8Array | null>(null);
  const [sourcePath, setSourcePath] = useState("");
  const [batchQueue, setBatchQueue] = useState<BatchItem[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  // 导出设置：按百分比缩放（10~200，100=原尺寸）+ 批量文件名后缀
  const [exportPercent, setExportPercent] = useState(100);
  const [exportSuffix, setExportSuffix] = useState("-cutout");

  const running = stage !== "idle" && stage !== "done";
  const busy = modelStatus !== "ready" || running || batchRunning;
  const downloadDisabled = !resultBytes || running || batchRunning;
  const hasImage = !!(originalUrl || resultUrl);

  // 用 ref 读取最新状态，避免拖放监听闭包拿到旧值
  const modelStatusRef = useRef(modelStatus);
  modelStatusRef.current = modelStatus;
  const hardEdgeRef = useRef(hardEdge);
  hardEdgeRef.current = hardEdge;
  const smoothEdgeRef = useRef(smoothEdge);
  smoothEdgeRef.current = smoothEdge;
  const exportPercentRef = useRef(exportPercent);
  exportPercentRef.current = exportPercent;
  const exportSuffixRef = useRef(exportSuffix);
  exportSuffixRef.current = exportSuffix;

  const currentBatchIdRef = useRef<string | null>(null);
  const lastDropTsRef = useRef(0);

  // 模型状态 + 管线阶段事件
  useEffect(() => {
    invoke<string>("get_model_status").then((s) => {
      if (s === "ready" || s === "ready:gpu" || s === "ready:cpu") {
        setModelStatus("ready");
        setBackend(s === "ready:gpu" ? "gpu" : "cpu");
      } else if (s.startsWith("error")) {
        setModelStatus("error");
        setModelError(s.slice(7));
      }
    });
    const unlistenStatus = listen<string>("model-status", (e) => {
      const s = e.payload;
      if (s === "ready" || s === "ready:gpu" || s === "ready:cpu") {
        setModelStatus("ready");
        setBackend(s === "ready:gpu" ? "gpu" : "cpu");
        setModelError("");
        setStatus(
          `模型已就绪，可以开始抠图（${s === "ready:gpu" ? "GPU 加速" : "CPU 模式"}）`,
        );
      } else if (s === "loading") {
        setModelStatus("loading");
      } else if (s.startsWith("error:")) {
        setModelStatus("error");
        setModelError(s.slice(7));
      }
    });
    const unlistenStage = listen<Stage>("pipeline-stage", (e) => {
      const s = e.payload;
      if (currentBatchIdRef.current) {
        // 批量模式：把阶段映射到当前项的进度
        const map: Record<string, number> = {
          inference: 35,
          postprocess: 65,
          encoding: 90,
          done: 100,
        };
        const prog = map[s];
        if (prog !== undefined) {
          setBatchQueue((q) =>
            q.map((x) =>
              x.id === currentBatchIdRef.current
                ? { ...x, progress: prog, status: s === "done" ? "done" : "running" }
                : x,
            ),
          );
        }
      } else {
        setStage(s);
        if (s === "done") setStatus("处理完成");
      }
    });
    return () => {
      void unlistenStatus.then((f) => f());
      void unlistenStage.then((f) => f());
    };
  }, []);

  // 系统级拖拽事件（拿到真实文件路径）—— 主链路
  useEffect(() => {
    const un = getCurrentWebview().onDragDropEvent((event) => {
      const p = event.payload;
      if (p.type === "enter" || p.type === "over") {
        setDragOver(true);
        return;
      }
      if (p.type === "drop") {
        setDragOver(false);
        const paths = p.paths ?? [];
        if (paths.length === 0) return;
        lastDropTsRef.current = Date.now();
        routeByCount(paths);
      } else {
        setDragOver(false);
      }
    });
    return () => {
      void un.then((f) => f());
    };
  }, []);

  const processFile = useCallback(async (path: string) => {
    const ext = extOf(path);
    if (!SUPPORTED.includes(ext)) {
      setStatus(`不支持的图片格式：.${ext}，请使用 PNG / JPG / JPEG / WebP / BMP`);
      return;
    }
    if (modelStatusRef.current !== "ready") {
      setStatus("模型尚未就绪，请等待加载完成");
      return;
    }
    setStage("inference");
    setSourcePath(path);
    setView("result");
    try {
      setStatus("读取图片...");
      const bytes = await invoke<number[]>("read_image", { path });
      const u8 = new Uint8Array(bytes);
      setOriginalUrl(bytesToDataUrl(u8));
      const png = await invoke<number[]>("segment_image", {
        imageBytes: bytes,
        hardEdge: hardEdgeRef.current,
        smooth: smoothEdgeRef.current,
      });
      const ru8 = new Uint8Array(png);
      setResultBytes(ru8);
      setResultUrl(bytesToDataUrl(ru8));
      setStatus(`完成：${baseName(path)}（${hardEdgeRef.current ? "硬边" : "软边"}）`);
      setStage("done");
    } catch (e) {
      setStage("idle");
      setStatus(`处理失败：${e}`);
    }
  }, []);

  // 取文件所在的目录（去掉最后的文件名）
  const dirOf = (p: string) => p.replace(/[\\/][^\\/]*$/, "");

  const enqueueBatch = useCallback(
    async (items: { path?: string; file?: File; name: string }[]) => {
      if (modelStatusRef.current !== "ready") {
        setStatus("模型尚未就绪，请等待加载完成");
        return;
      }
      if (items.length === 0) return;

      // 没有文件路径（纯 HTML5 拖入）时，回退到"下载"目录保存
      let downloads = "";
      if (items.some((it) => !it.path)) {
        try {
          downloads = (await downloadDir()) ?? "";
        } catch {
          downloads = "";
        }
      }

      const queue: BatchItem[] = items.map((it) => ({
        id: uid(),
        path: it.path,
        file: it.file,
        name: it.name,
        status: "pending",
        progress: 0,
      }));
      setBatchQueue(queue);
      setBatchRunning(true);
      setStatus(`批量处理：0/${queue.length}`);

      for (let i = 0; i < queue.length; i++) {
        const item = queue[i];
        currentBatchIdRef.current = item.id;
        setBatchQueue((q) =>
          q.map((x) => (x.id === item.id ? { ...x, status: "running", progress: 10 } : x)),
        );
        try {
          let bytes: Uint8Array;
          if (item.path) {
            bytes = new Uint8Array(await invoke<number[]>("read_image", { path: item.path }));
          } else if (item.file) {
            bytes = new Uint8Array(await item.file.arrayBuffer());
          } else {
            throw new Error("缺少文件来源");
          }
          const png = await invoke<number[]>("segment_image", {
            imageBytes: Array.from(bytes),
            hardEdge: hardEdgeRef.current,
            smooth: smoothEdgeRef.current,
          });
          // 按百分比缩放导出（100 = 原尺寸，跳过缩放）
          let outBytes: number[] = png;
          if (exportPercentRef.current !== 100) {
            outBytes = await invoke<number[]>("resize_png", {
              pngBytes: png,
              percent: exportPercentRef.current,
            });
          }
          // 自动保存到：有原始路径则存回原图所在文件夹；否则存到下载目录
          // 结果永远导出为透明 PNG（.jpg 无法保存透明通道）
          const outDir = item.path ? dirOf(item.path) : downloads;
          const outName = cutoutName(item.name, exportSuffixRef.current || "-cutout");
          const outPath = `${outDir}/${outName}`;
          await invoke("save_png", { bytes: outBytes, path: outPath });
          setBatchQueue((q) =>
            q.map((x) => (x.id === item.id ? { ...x, status: "done", progress: 100 } : x)),
          );
        } catch (e) {
          setBatchQueue((q) =>
            q.map((x) =>
              x.id === item.id
                ? { ...x, status: "error", progress: 0, error: String(e) }
                : x,
            ),
          );
        }
        setStatus(`批量处理：${i + 1}/${queue.length}`);
      }
      currentBatchIdRef.current = null;
      setBatchRunning(false);
      const done = queue.filter((x) => x.status === "done").length;
      const folderHint = items.some((it) => !it.path)
        ? "（无路径的图已存到下载目录）"
        : "（已存回原图所在文件夹）";
      setStatus(`批量完成：${done}/${queue.length} 张${folderHint}`);
    },
    [],
  );

  const routeByCount = useCallback(
    (paths: string[]) => {
      // 需求 1：单张图片不能在界面外拖入，只能单击操作区选择；
      // 拖入只接受"多张"（即需求 5 的批量拖入）
      if (paths.length === 1) {
        setStatus("单张图片请直接单击操作区选择；如需批量，请一次性拖入多张图片");
        return;
      }
      enqueueBatch(paths.map((p) => ({ path: p, name: baseName(p) })));
    },
    [enqueueBatch],
  );

  const routeFilesByCount = useCallback(
    (files: File[]) => {
      // 同上：单张不允许拖入，只走单击；此处兜底仅在 Tauri 未拦截时触发
      if (files.length === 1) {
        setStatus("单张图片请直接单击操作区选择；如需批量，请一次性拖入多张图片");
        return;
      }
      enqueueBatch(files.map((f) => ({ file: f, name: f.name })));
    },
    [enqueueBatch],
  );

  const pickFile = useCallback(async () => {
    if (busy) return;
    const path = await open({
      multiple: false,
      filters: [{ name: "图片", extensions: SUPPORTED }],
    });
    if (typeof path === "string") processFile(path);
  }, [busy, processFile]);

  const download = useCallback(async () => {
    if (!resultBytes) return;
    const dest = await save({
      title: "保存透明 PNG（可在弹窗中修改文件名）",
      defaultPath: sourcePath ? cutoutName(sourcePath, exportSuffixRef.current || "-cutout") : "cutout.png",
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (!dest) return;
    try {
      let bytes = Array.from(resultBytes);
      if (exportPercentRef.current !== 100) {
        bytes = await invoke<number[]>("resize_png", {
          pngBytes: bytes,
          percent: exportPercentRef.current,
        });
      }
      await invoke("save_png", { bytes, path: dest });
      setStatus(
        `已保存：${dest}${exportPercentRef.current !== 100 ? `（尺寸 ${exportPercentRef.current}%）` : ""}`,
      );
    } catch (e) {
      setStatus(`保存失败：${e}`);
    }
  }, [resultBytes, sourcePath]);

  // 清空批量列表（清屏）：结果保留，仅移除列表记录
  const clearBatchList = useCallback(() => {
    setBatchQueue([]);
    setStatus("");
  }, []);

  // 批量处理：按需求 3，只选"文件夹"，自动处理其中所有支持的图片
  const batchProcess = useCallback(async () => {
    if (busy) return;
    const dir = await open({ directory: true, title: "选择包含图片的文件夹" });
    if (typeof dir !== "string") return;
    const names = await invoke<string[]>("list_image_files", { dir });
    if (!names || names.length === 0) {
      setStatus("该文件夹内没有支持的图片（PNG / JPG / JPEG / WebP / BMP）");
      return;
    }
    enqueueBatch(names.map((n) => ({ path: `${dir}/${n}`, name: n })));
  }, [busy, enqueueBatch]);

  // 魔法橡皮擦（需求 9）：点击处开始，清除连在一起的同色/相近色前景
  const handleErase = useCallback(
    async (p: ErasePoint) => {
      if (!resultBytes) return;
      try {
        const png = await invoke<number[]>("erase_region", {
          imageBytes: Array.from(resultBytes),
          x: p.x,
          y: p.y,
          tolerance,
        });
        const ru8 = new Uint8Array(png);
        setResultBytes(ru8);
        setResultUrl(bytesToDataUrl(ru8));
        setStatus("已擦除一块连通区域，可继续点击擦除；擦错请点右上角 × 重来");
      } catch (e) {
        setStatus(`擦除失败：${e}`);
      }
    },
    [resultBytes, tolerance],
  );

  // 清空结果，回到初始态
  const closeResult = useCallback(() => {
    setResultBytes(null);
    setResultUrl(null);
    setOriginalUrl(null);
    setSourcePath("");
    setStage("idle");
    setView("result");
    setEraseMode(false);
    setStatus("");
  }, []);

  const reloadModel = useCallback(async () => {
    setModelStatus("loading");
    try {
      await invoke("reload_model");
    } catch (e) {
      setModelError(String(e));
    }
  }, []);

  // HTML5 拖放兜底（仅当 Tauri 未抢先拦截时生效）
  const onHtml5Drop = useCallback(
    (e: ReactDragEvent) => {
      e.preventDefault();
      const files = e.dataTransfer?.files;
      if (!files || files.length === 0) return;
      if (Date.now() - lastDropTsRef.current < 600) return; // Tauri 已处理过
      routeFilesByCount(Array.from(files));
    },
    [routeFilesByCount],
  );

  return (
    <main className="container">
      <header className="header">
        <h1>一键抠图 · 硬边输出</h1>
        <div className="header-row">
          <label className="toggle" title="不勾选 = 平滑羽化边缘（默认，接近 PS 钢笔级干净轮廓）；勾选 = 纯硬边 0/255 二值">
            <input
              type="checkbox"
              checked={hardEdge}
              onChange={(e) => setHardEdge(e.target.checked)}
              disabled={running || batchRunning}
            />
            硬边模式
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={smoothEdge}
              onChange={(e) => setSmoothEdge(e.target.checked)}
              disabled={running || batchRunning}
            />
            边缘平滑
          </label>
          <label className="toggle" title="开启后优先使用 GPU 加速（Windows: DirectML，macOS: CoreML），不支持的算子自动回退 CPU，即 CPU+GPU 混合执行；无显卡或驱动异常时自动以 CPU 运行">
            <input
              type="checkbox"
              checked={gpuAccel}
              onChange={async (e) => {
                const v = e.target.checked;
                setGpuAccel(v);
                try {
                  await invoke("reload_model", { useGpu: v });
                } catch (err) {
                  setModelError(String(err));
                }
              }}
              disabled={modelStatus !== "ready" || running || batchRunning}
            />
            GPU 加速
          </label>
          <div className="view-toggle">
            <button
              className={view === "original" ? "active" : ""}
              onClick={() => setView("original")}
              disabled={!originalUrl}
            >
              原图
            </button>
            <button
              className={view === "result" ? "active" : ""}
              onClick={() => setView("result")}
              disabled={!resultUrl}
            >
              结果
            </button>
          </div>
          <span className={`model-badge ${modelStatus}`}>
            {modelStatus === "loading"
              ? "模型加载中"
              : modelStatus === "ready"
                ? `模型就绪 · ${backend === "gpu" ? "GPU" : "CPU"}`
                : "模型加载失败"}
          </span>
          {resultUrl && (
            <button
              className={`eraser-btn${eraseMode ? " active" : ""}`}
              title="魔法橡皮擦：点击图中颜色，擦除连在一起的同色/相近色区域"
              onClick={() => {
                setEraseMode((m) => !m);
                setView("result");
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="m2 22 1-1h3l9-9" />
                <path d="M3 21v-3l9-9" />
                <path d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3l.4.4Z" />
              </svg>
              魔法橡皮擦
            </button>
          )}
        </div>
        {resultUrl && eraseMode && (
          <div className="tolerance-row">
            <span>容差（越大擦得越多）</span>
            <input
              type="range"
              min={5}
              max={100}
              value={tolerance}
              onChange={(e) => setTolerance(Number(e.target.value))}
            />
            <span className="tolerance-value">{tolerance}</span>
          </div>
        )}
      </header>

      <section
        className={`dropzone${dragOver ? " dragover" : ""}${busy ? " busy" : ""}`}
        onClick={busy || hasImage ? undefined : pickFile}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragOver(false)}
        onDrop={onHtml5Drop}
      >
        {modelStatus === "loading" ? (
          <p>模型加载中，请稍候...</p>
        ) : modelStatus === "error" ? (
          <div className="model-error">
            <p>模型加载失败：{modelError}</p>
            <button className="test-btn" onClick={reloadModel}>
              重新加载
            </button>
          </div>
        ) : running ? (
          <div className="progress-box">
            <div className="spinner" />
            <p>
              {stage === "inference"
                ? "推理中（GPU 约 2~10 秒 / CPU 约 20~35 秒）..."
                : stage === "postprocess"
                  ? "后处理中..."
                  : "编码中..."}
            </p>
          </div>
        ) : batchRunning || batchQueue.length > 0 ? (
          <BatchList items={batchQueue} />
        ) : originalUrl || resultUrl ? (
          <PreviewView
            src={view === "original" ? originalUrl! : resultUrl!}
            alt={view === "original" ? "原图" : "抠图结果"}
            mode={eraseMode && view === "result" ? "erase" : "none"}
            onEraseClick={handleErase}
            onRequestPick={pickFile}
            onClose={resultUrl ? closeResult : undefined}
          />
        ) : (
          <p>单击选择单张图片，或一次性拖入多张批量处理；也可点「批量处理」选文件夹</p>
        )}
      </section>

      {status && (
        <p
          className={`status${
            status.startsWith("处理失败") ||
            status.startsWith("保存失败") ||
            status.includes("不支持的") ||
            status.includes("失败")
              ? " error"
              : ""
          }`}
        >
          {status}
        </p>
      )}

      <footer className="footer">
        <div className="export-row" title="导出设置">
          <label>
            导出尺寸
            <input
              type="number"
              min={10}
              max={200}
              step={5}
              value={exportPercent}
              onChange={(e) => {
                const v = Number(e.target.value);
                setExportPercent(Number.isNaN(v) ? 100 : Math.min(200, Math.max(10, v)));
              }}
              disabled={batchRunning}
            />
            %
          </label>
          <label title="批量导出时的文件名后缀，如 photo.jpg -> photo-cutout.png">
            批量文件名后缀
            <input
              type="text"
              value={exportSuffix}
              onChange={(e) => setExportSuffix(e.target.value)}
              placeholder="-cutout"
              disabled={batchRunning}
            />
          </label>
        </div>
        <div className="footer-btns">
          <button className="download-btn" onClick={download} disabled={downloadDisabled}>
            下载透明 PNG
          </button>
          <button className="test-btn" onClick={batchProcess} disabled={busy}>
            批量处理
          </button>
          {batchQueue.length > 0 && !batchRunning && (
            <button className="test-btn" onClick={clearBatchList} title="清空批量列表记录">
              清空列表
            </button>
          )}
        </div>
      </footer>
    </main>
  );
}

export default App;
