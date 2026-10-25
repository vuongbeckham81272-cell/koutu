import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent as ReactDragEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import PreviewView from "./components/PreviewView";
import BatchList from "./components/BatchList";
import type { BatchItem, ModelStatus, Stage } from "./types";
import "./App.css";

const SUPPORTED = ["png", "jpg", "jpeg", "webp", "bmp"];

function extOf(path: string) {
  return path.split(".").pop()?.toLowerCase() ?? "";
}

function baseName(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

function cutoutName(path: string) {
  const name = baseName(path);
  const dot = name.lastIndexOf(".");
  return dot > 0 ? `${name.slice(0, dot)}-cutout${name.slice(dot)}` : `${name}-cutout.png`;
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
  const [stage, setStage] = useState<Stage>("idle");
  const [status, setStatus] = useState("");
  const [hardEdge, setHardEdge] = useState(true);
  const [view, setView] = useState<"original" | "result">("result");
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultBytes, setResultBytes] = useState<Uint8Array | null>(null);
  const [sourcePath, setSourcePath] = useState("");
  const [batchQueue, setBatchQueue] = useState<BatchItem[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);

  const running = stage !== "idle" && stage !== "done";
  const busy = modelStatus !== "ready" || running || batchRunning;
  const downloadDisabled = !resultBytes || running || batchRunning;
  const hasImage = !!(originalUrl || resultUrl);

  // 用 ref 读取最新状态，避免拖放监听闭包拿到旧值
  const modelStatusRef = useRef(modelStatus);
  modelStatusRef.current = modelStatus;
  const hardEdgeRef = useRef(hardEdge);
  hardEdgeRef.current = hardEdge;

  const currentBatchIdRef = useRef<string | null>(null);
  const lastDropTsRef = useRef(0);

  // 模型状态 + 管线阶段事件
  useEffect(() => {
    invoke<string>("get_model_status").then((s) => {
      if (s === "ready") setModelStatus("ready");
      else if (s.startsWith("error")) {
        setModelStatus("error");
        setModelError(s.slice(7));
      }
    });
    const unlistenStatus = listen<string>("model-status", (e) => {
      const s = e.payload;
      if (s === "ready") {
        setModelStatus("ready");
        setModelError("");
        setStatus("模型已就绪，可以开始抠图");
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

  // 从字节直接处理（HTML5 拖放兜底，无路径）
  const processBytes = useCallback(async (bytes: Uint8Array, name: string) => {
    if (modelStatusRef.current !== "ready") {
      setStatus("模型尚未就绪，请等待加载完成");
      return;
    }
    setStage("inference");
    setSourcePath("");
    setView("result");
    try {
      setStatus("处理中...");
      setOriginalUrl(bytesToDataUrl(bytes));
      const png = await invoke<number[]>("segment_image", {
        imageBytes: Array.from(bytes),
        hardEdge: hardEdgeRef.current,
      });
      const ru8 = new Uint8Array(png);
      setResultBytes(ru8);
      setResultUrl(bytesToDataUrl(ru8));
      setStatus(`完成：${name}`);
      setStage("done");
    } catch (e) {
      setStage("idle");
      setStatus(`处理失败：${e}`);
    }
  }, []);

  const enqueueBatch = useCallback(
    async (items: { path?: string; file?: File; name: string }[]) => {
      if (modelStatusRef.current !== "ready") {
        setStatus("模型尚未就绪，请等待加载完成");
        return;
      }
      if (items.length === 0) return;

      const outputDir = await open({ directory: true, title: "选择输出文件夹" });
      if (typeof outputDir !== "string") return;

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
          });
          const outPath = `${outputDir}/${cutoutName(item.name)}`;
          await invoke("save_png", { bytes: Array.from(png), path: outPath });
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
      setStatus(
        `批量完成：${queue.filter((x) => x.status === "done").length}/${queue.length} 张 → ${outputDir}`,
      );
    },
    [],
  );

  const routeByCount = useCallback(
    (paths: string[]) => {
      if (paths.length === 1) processFile(paths[0]);
      else enqueueBatch(paths.map((p) => ({ path: p, name: baseName(p) })));
    },
    [processFile, enqueueBatch],
  );

  const routeFilesByCount = useCallback(
    (files: File[]) => {
      if (files.length === 1) {
        files[0]
          .arrayBuffer()
          .then((buf) => processBytes(new Uint8Array(buf), files[0].name));
      } else {
        enqueueBatch(files.map((f) => ({ file: f, name: f.name })));
      }
    },
    [processBytes, enqueueBatch],
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
      title: "保存透明 PNG",
      defaultPath: sourcePath ? cutoutName(sourcePath) : "cutout.png",
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (!dest) return;
    try {
      await invoke("save_png", { bytes: Array.from(resultBytes), path: dest });
      setStatus(`已保存：${dest}`);
    } catch (e) {
      setStatus(`保存失败：${e}`);
    }
  }, [resultBytes, sourcePath]);

  // 批量处理：选多张图片文件（不再选文件夹）
  const batchProcess = useCallback(async () => {
    if (busy) return;
    const paths = await open({
      multiple: true,
      filters: [{ name: "图片", extensions: SUPPORTED }],
    });
    if (!paths || paths.length === 0) return;
    enqueueBatch(paths.map((p) => ({ path: p, name: baseName(p) })));
  }, [busy, enqueueBatch]);

  // 清空结果，回到初始态
  const closeResult = useCallback(() => {
    setResultBytes(null);
    setResultUrl(null);
    setOriginalUrl(null);
    setSourcePath("");
    setStage("idle");
    setView("result");
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
          <label className="toggle">
            <input
              type="checkbox"
              checked={hardEdge}
              onChange={(e) => setHardEdge(e.target.checked)}
              disabled={running || batchRunning}
            />
            硬边模式
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
                ? "模型就绪"
                : "模型加载失败"}
          </span>
        </div>
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
                ? "推理中（约 8~15 秒）..."
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
            onRequestPick={pickFile}
            onClose={resultUrl ? closeResult : undefined}
          />
        ) : (
          <p>把图片拖到这里，或双击选择文件</p>
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
        <button className="download-btn" onClick={download} disabled={downloadDisabled}>
          下载透明 PNG
        </button>
        <button className="test-btn" onClick={batchProcess} disabled={busy}>
          批量处理
        </button>
      </footer>
    </main>
  );
}

export default App;
