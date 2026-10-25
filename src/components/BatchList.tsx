import type { BatchItem, BatchStatus } from "../types";

function stateLabel(s: BatchStatus): string {
  switch (s) {
    case "pending":
      return "等待";
    case "running":
      return "处理中";
    case "done":
      return "完成";
    case "error":
      return "失败";
  }
}

export default function BatchList({ items }: { items: BatchItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="batch-list">
      {items.map((it) => (
        <div className="batch-item" key={it.id}>
          <span className="batch-name" title={it.name}>
            {it.name}
          </span>
          <span className={`batch-state ${it.status}`}>
            {it.status === "error" && it.error ? `失败: ${it.error.slice(0, 20)}` : stateLabel(it.status)}
          </span>
          <div className="batch-bar">
            <div className="batch-fill" style={{ width: `${it.progress}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
