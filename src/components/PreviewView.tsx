import { useEffect, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ErasePoint, Transform } from "../types";

interface PreviewViewProps {
  src: string;
  alt: string;
  /** 第 9 项预留：本次恒为 "none" */
  mode?: "none" | "erase";
  /** 双击图片 -> 打开文件选择窗口 */
  onRequestPick?: () => void;
  /** 第 9 项预留：橡皮擦点击回调 */
  onEraseClick?: (p: ErasePoint) => void;
  /** 关闭按钮（清空结果回到初始态） */
  onClose?: () => void;
}

const MIN_SCALE = 0.1;
const MAX_SCALE = 12;

export default function PreviewView({
  src,
  alt,
  mode = "none",
  onRequestPick,
  onEraseClick,
  onClose,
}: PreviewViewProps) {
  const [transform, setTransform] = useState<Transform>({ scale: 1, tx: 0, ty: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const downPosRef = useRef<{ x: number; y: number } | null>(null);
  const layoutRef = useRef<{ iw: number; ih: number }>({ iw: 0, ih: 0 });
  // 单击计数（用于双击选文件 / 三击复位视图）
  const clickCountRef = useRef(0);
  const clickTimerRef = useRef<number | null>(null);

  // 切换图片时重置变换与点击计数
  useEffect(() => {
    setTransform({ scale: 1, tx: 0, ty: 0 });
    clickCountRef.current = 0;
    if (clickTimerRef.current !== null) {
      clearTimeout(clickTimerRef.current);
      clickTimerRef.current = null;
    }
  }, [src]);

  // 滚轮缩放（React onWheel 是 passive，需要用 ref 自行挂载才能 preventDefault）
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      setTransform((t) => {
        const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, t.scale * factor));
        return { ...t, scale: next };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // 左键拖拽平移（监听挂到 window，避免拖出容器丢失）
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      setTransform((t) => ({ ...t, tx: d.tx + (e.clientX - d.x), ty: d.ty + (e.clientY - d.y) }));
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const onMouseDown = (e: ReactMouseEvent) => {
    if (e.button !== 0) return; // 仅左键
    downPosRef.current = { x: e.clientX, y: e.clientY };
    dragRef.current = { x: e.clientX, y: e.clientY, tx: transform.tx, ty: transform.ty };
  };

  // 单击计数（延迟 350ms 判定，兼容双击与三击两种手势）：
  //   双击（350ms 内点 2 次）-> 打开文件选择窗口
  //   三击（350ms 内点 3 次）-> 复位视图到刚导入的状态（缩放 100%、居中），不弹文件框
  const registerClick = () => {
    clickCountRef.current += 1;
    if (clickTimerRef.current !== null) clearTimeout(clickTimerRef.current);
    clickTimerRef.current = window.setTimeout(() => {
      const c = clickCountRef.current;
      clickCountRef.current = 0;
      clickTimerRef.current = null;
      if (c >= 3) {
        setTransform({ scale: 1, tx: 0, ty: 0 });
      } else if (c === 2) {
        onRequestPick?.();
      }
    }, 350);
  };

  const onClick = (e: ReactMouseEvent) => {
    // 拖拽平移后的松手不算"点击"：位移超过 4px 忽略
    const d = downPosRef.current;
    if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return;
    if (mode === "erase") {
      const p = toImageCoords(e.clientX, e.clientY);
      if (p) onEraseClick?.(p);
      return;
    }
    registerClick();
  };

  // 屏幕坐标 -> 原图像素坐标（为第 9 项橡皮擦预留，本轮已可用）
  function toImageCoords(clientX: number, clientY: number): ErasePoint | null {
    const rect = containerRef.current?.getBoundingClientRect();
    const { iw, ih } = layoutRef.current;
    if (!rect || !iw || !ih) return null;
    const W = rect.width;
    const H = rect.height;
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;

    const fit = Math.min(W / iw, H / ih); // contain 适配系数
    const baseW = iw * fit;
    const baseH = ih * fit;
    const baseX = (W - baseW) / 2;
    const baseY = (H - baseH) / 2;

    const centerX = baseX + baseW / 2 + transform.tx;
    const centerY = baseY + baseH / 2 + transform.ty;

    const imgX = ((cx - centerX) / transform.scale + baseW / 2) / fit;
    const imgY = ((cy - centerY) / transform.scale + baseH / 2) / fit;
    return { x: Math.round(imgX), y: Math.round(imgY), naturalWidth: iw, naturalHeight: ih };
  }

  return (
    <div
      className={`preview-wrap${mode === "erase" ? " erasing" : ""}`}
      ref={containerRef}
    >
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        className="preview-img"
        draggable={false}
        onMouseDown={onMouseDown}
        onClick={onClick}
        onLoad={(e) => {
          layoutRef.current = {
            iw: e.currentTarget.naturalWidth,
            ih: e.currentTarget.naturalHeight,
          };
        }}
        style={{
          transform: `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`,
        }}
      />
      {onClose && (
        <button className="close-btn" onClick={onClose} title="关闭 / 返回">
          ×
        </button>
      )}
    </div>
  );
}
