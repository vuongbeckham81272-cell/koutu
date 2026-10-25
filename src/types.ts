export type ModelStatus = "loading" | "ready" | "error";
export type Stage = "idle" | "inference" | "postprocess" | "encoding" | "done";

export interface Transform {
  scale: number;
  tx: number;
  ty: number;
}

export type BatchStatus = "pending" | "running" | "done" | "error";
export interface BatchItem {
  id: string;
  path?: string; // Tauri 文件路径（主链路）
  file?: File; // HTML5 兜底：拖入时取不到路径，用 File.arrayBuffer()
  name: string;
  status: BatchStatus;
  progress: number; // 0~100
  error?: string;
}

// 第 9 项魔术橡皮擦预留参数（本轮仅定义，不消费）
export interface ErasePoint {
  x: number; // 原图像素 X
  y: number; // 原图像素 Y
  naturalWidth: number;
  naturalHeight: number;
}
