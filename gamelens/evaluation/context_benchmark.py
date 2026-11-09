"""Offline structural-context size comparison, not measured token/latency gains."""
import argparse
import json
from pathlib import Path

from gamelens.core.paths import PROJECT_ROOT
from gamelens.core.runtime import PlannerDecision, planner_prompt
from gamelens.evaluation.suite import business_registry
from gamelens.memory.context import PlannerContextBuilder, compact_schema
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text


def measure_messages(messages):
    """Measure the exact serialized messages without pretending bytes are tokens."""
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return {"characters": len(serialized), "utf8_bytes": len(serialized.encode("utf-8")),
            "exact_tokens": None, "measurement": "serialized_messages_utf8"}


def planner_messages(state, tools, *, compact):
    builder = PlannerContextBuilder()
    context = builder.build(state, tools, compact=compact)
    schema = PlannerDecision.model_json_schema()
    if state.get("investigation_profile") == "focused":
        schema["properties"].pop("version_evidence", None)
        schema.get("$defs", {}).pop("VersionEvidenceAssignment", None)
    if compact:
        context["planner_output_schema"] = compact_schema(schema, context.setdefault("$defs", {}))
    else:
        context["planner_output_schema"] = schema
    return [{"role": "system", "content": planner_prompt(state.get("investigation_profile", "extended"))},
            {"role": "user", "content": "规划上下文（已包含可用工具）：\n" +
             json.dumps(context, ensure_ascii=False)}]


def main():
    parser = argparse.ArgumentParser(description="无模型的末轮结构上下文体积对照")
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evals/results/context_budget.json")
    args = parser.parse_args()
    tools = business_registry(PROJECT_ROOT, None).specifications()
    rows = []
    for path in args.run:
        state = json.loads(path.read_text(encoding="utf-8"))
        reference = measure_messages(planner_messages(state, tools, compact=False))
        compact = measure_messages(planner_messages(state, tools, compact=True))
        rows.append({"run": path.stem, "reference_chars": reference["characters"],
                     "compact_chars": compact["characters"],
                     "reference_utf8_bytes": reference["utf8_bytes"],
                     "compact_utf8_bytes": compact["utf8_bytes"],
                     "reduction": (reference["utf8_bytes"]-compact["utf8_bytes"])/reference["utf8_bytes"],
                     "exact_tokens": None})
    payload = {"rows": rows, "catalog": "首轮固定资料评测的8种工具", "model_calls": 0,
               "measurement": "完整system/user消息序列化后的UTF-8字节数",
               "note": "从保存状态重建同一工具目录的完整规划消息，包含系统提示与输出schema；字节数不是供应商token或速度测量。"}
    atomic_write_json(args.output, payload)
    lines = ["# 结构上下文体积对照（离线）", "", payload["note"], "",
             "| 记录 | 原消息字节数 | 压缩后字节数 | 减少 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['reference_utf8_bytes']} | {row['compact_utf8_bytes']} | {row['reduction']:.1%} |")
    lines += ["", "完整工具观察、来源哈希和原文保留在状态中；只缩减发给模型的投影。所有工具仍可见，参数约束保留。", "",
              "真实调用次数与耗时需要在明确允许费用后，使用full/hybrid模式分别复测，不能把字符减少直接换算成速度或费用收益。"]
    atomic_write_text(args.output.with_suffix(".md"), "\n".join(lines)+"\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
