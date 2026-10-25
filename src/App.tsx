import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./App.css";

const SUPPORTED = ["png", "jpg", "jpeg", "webp", "bmp"];

type ModelStatus = "loading" | "ready" | "error";
type Stage = "idle" | "inference" | "postprocess" | "encoding" | "done";

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
  const [batchProgress, setBatchProgress] = useState("");
  const busy = modelStatus !== "ready" || (stage !== "idle" && stage !== "done");
  const running = stage !== "idle" && stage !== "done";

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
      setStage(e.payload);
      if (e.payload === "done") setStatus("处理完成");
    });
    return () => {
      void unlistenStatus.then((f) => f());
      void unlistenStage.then((f) => f());
    };
  }, []);

  // 系统级拖拽事件（拿到真实文件路径）
  useEffect(() => {
    const un = getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "enter" || event.payload.type === "over") {
        setDragOver(true);
      } else if (event.payload.type === "drop") {
        setDragOver(false);
        const paths = (event.payload as { paths?: string[] }).paths ?? [];
        if (paths.length > 0) processFile(paths[0]);
      } else {
        setDragOver(false);
      }
    });
    return () => {
      void un.then((f) => f());
    };
  }, []);

  const processFile = useCallback(
    async (path: string) => {
      const ext = extOf(path);
      if (!SUPPORTED.includes(ext)) {
        setStatus(`不支持的图片格式：.${ext}，请使用 PNG / JPG / JPEG / WebP / BMP`);
        return;
      }
      if (modelStatus !== "ready") {
        setStatus("模型尚未就绪，请等待加载完成");
        return;
      }
      setStage("inference");
      setSourcePath(path);
      try {
        setStatus("读取图片...");
        const bytes = await invoke<number[]>("read_image", { path });
        setOriginalUrl(bytesToDataUrl(new Uint8Array(bytes)));
        const png = await invoke<number[]>("segment_image", {
          imageBytes: bytes,
          hardEdge,
        });
        const u8 = new Uint8Array(png);
        setResultBytes(u8);
        setResultUrl(bytesToDataUrl(u8));
        setView("result");
        setStatus(`完成：${baseName(path)}（${hardEdge ? "硬边" : "软边"}）`);
        setStage("done");
      } catch (e) {
        setStage("idle");
        setStatus(`处理失败：${e}`);
      }
    },
    [hardEdge, modelStatus],
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

  // 批量处理：选文件夹 -> 逐张处理 -> 保存到输出文件夹
  const batchProcess = useCallback(async () => {
    if (busy) return;
    const inputDir = await open({ directory: true, title: "选择包含图片的文件夹" });
    if (typeof inputDir !== "string") return;
    const outputDir = await open({ directory: true, title: "选择输出文件夹" });
    if (typeof outputDir !== "string") return;

    let names: string[];
    try {
      names = await invoke<string[]>("list_image_files", { dir: inputDir });
    } catch (e) {
      setStatus(`读取文件夹失败：${e}`);
      return;
    }
    if (names.length === 0) {
      setStatus(`文件夹中没有受支持的图片（${SUPPORTED.join(" / ")}）`);
      return;
    }
    setStage("inference");
    let ok = 0;
    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      setBatchProgress(`批量处理：${i + 1}/${names.length} ${name}`);
      try {
        const bytes = await invoke<number[]>("read_image", { path: `${inputDir}/${name}` });
        const png = await invoke<number[]>("segment_image", {
          imageBytes: bytes,
          hardEdge,
        });
        const outName = cutoutName(name);
        await invoke("save_png", {
          bytes: Array.from(png),
          path: `${outputDir}/${outName}`,
        });
        ok++;
      } catch (e) {
        setStatus(`「${name}」处理失败：${e}`);
      }
    }
    setBatchProgress("");
    setStage("done");
    setStatus(`批量完成：${ok}/${names.length} 张已保存到 ${outputDir}`);
  }, [busy, hardEdge]);

  const reloadModel = useCallback(async () => {
    setModelStatus("loading");
    try {
      await invoke("reload_model");
    } catch (e) {
      setModelError(String(e));
    }
  }, []);

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
              disabled={running}
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
        onClick={busy ? undefined : pickFile}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragOver(false)}
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
        ) : originalUrl || resultUrl ? (
          <img
            src={view === "original" ? originalUrl! : resultUrl!}
            alt={view === "original" ? "原图" : "抠图结果"}
            className="preview"
          />
        ) : (
          <p>把图片拖到这里，或点击选择文件</p>
        )}
      </section>

      {(status || batchProgress) && (
        <p className={`status${status.startsWith("处理失败") || status.startsWith("保存失败") || status.includes("不支持的") || status.includes("失败") ? " error" : ""}`}>
          {batchProgress || status}
        </p>
      )}

      <footer className="footer">
        <button
          className="download-btn"
          onClick={download}
          disabled={!resultBytes || running}
        >
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
