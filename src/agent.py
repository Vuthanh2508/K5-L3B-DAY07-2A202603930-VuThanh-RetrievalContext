"""Agent Implementation for Day 7 — Agentic RAG.
Implements Controller Protocol using OpenAI GPT-4o-mini with native Tool Calling,
connecting to the Day 7 Retriever and conformant with starter/agentic.py.
"""
# CRITICAL FOR WINDOWS: torch and FlagEmbedding must be imported BEFORE openai/httpx to prevent OpenMP runtime DLL crash
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

from FlagEmbedding import BGEM3FlagModel
from retrieval import ZScoreHybridRetriever, DenseRetriever, BM25Retriever
from openai import OpenAI
from agentic import run_agent

import argparse
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional


class OpenAIController:
    """LLM Controller using OpenAI Tool Calling (gpt-4o-mini)."""
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Tìm kiếm các đoạn văn bản (chunks) liên quan trong báo cáo tài chính Form 10-Q Q2/2023 của Netflix.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Câu truy vấn tìm kiếm bằng tiếng Việt hoặc tiếng Anh (từ khóa tài chính cụ thể)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        system_msg = (
            "Bạn là trợ lý AI chuyên gia phân tích báo cáo tài chính SEC.\n"
            "TÀI LIỆU BÁO CÁO TÀI CHÍNH GỐC HOÀN TOÀN BẰNG TIẾNG ANH.\n"
            "Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên các đoạn trích tài liệu được cung cấp.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Nếu chưa có ngữ cảnh tìm kiếm, hoặc ngữ cảnh hiện tại chưa đủ để trả lời câu hỏi, HÃY GỌI TOOL `search(query=...)`.\n"
            "   QUY TẮC TÌM KIẾM QUAN TRỌNG (Cross-lingual Query Transformation & Chống trùng lặp): Vì toàn bộ tài liệu được viết bằng TIẾNG ANH, khi gọi tool `search`, bạn BẮT BUỘC PHẢI DỊCH VÀ CHUYỂN ĐỔI câu truy vấn sang TIẾNG ANH CHUYÊN NGÀNH TÀI CHÍNH với các từ khóa cô đọng, chính xác. TUYỆT ĐỐI KHÔNG lặp lại các câu truy vấn tìm kiếm giống hệt nhau qua các lượt gọi.\n"
            "2. Khi đã tìm được đoạn trích có chứa bằng chứng, hãy trả lời bằng TIẾNG VIỆT chính xác, ngắn gọn và BẮT BUỘC TRÍCH DẪN mã chunk ở định dạng `[chunk-id]` ngay trong câu trả lời.\n"
            "3. Nếu sau khi tìm kiếm (tối đa các lượt) mà tài liệu KHÔNG có thông tin, hoặc câu hỏi hỏi về thông tin ngoài phạm vi tài liệu (như số liệu năm khác, chỉ số không công bố, chi tiết không có trong 10-Q), bạn PHẢI NÓI RÕ: 'Không đủ bằng chứng để trả lời.' và TUYỆT ĐỐI KHÔNG bịa đặt số liệu hay trích dẫn chunk không liên quan.\n"
            "4. Tuyệt đối chỉ trích dẫn mã chunk xuất hiện trong ngữ cảnh được cung cấp."
        )

        messages = [{"role": "system", "content": system_msg}]

        if not history:
            messages.append({
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nHiện tại chưa có đoạn trích nào. Hãy gọi tool `search` với từ khóa tiếng Anh tương ứng để tìm kiếm tài liệu."
            })
        else:
            latest = history[-1]
            ctx = latest.get("context", "")
            queries_so_far = latest.get("queries", [])
            messages.append({
                "role": "user",
                "content": (
                    f"Câu hỏi gốc: {question}\n\n"
                    f"Các truy vấn tìm kiếm tiếng Anh đã thực hiện: {queries_so_far}\n\n"
                    f"Ngữ cảnh tài liệu tích lũy hiện có:\n{ctx}\n\n"
                    "Hãy kiểm tra kỹ: Nếu ngữ cảnh trên đã đủ thông tin, hãy đưa ra câu trả lời bằng ngôn ngữ của câu hỏi gốc kèm citation `[chunk-id]`.\n"
                    "Nếu thông tin chưa đủ nhưng có thể tìm thấy trong 10-Q, hãy gọi tiếp tool `search` với từ khóa tiếng Anh mới (LƯU Ý: TUYỆT ĐỐI KHÔNG lặp lại bất kỳ truy vấn nào đã có trong danh sách truy vấn đã thực hiện ở trên, hãy đổi từ đồng nghĩa hoặc góc độ tiếp cận khác).\n"
                    "Nếu tài liệu hoàn toàn không có thông tin, hãy trả lời 'Không đủ bằng chứng để trả lời.'"
                )
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )

        choice = response.choices[0]
        msg = choice.message
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            if tc.function.name == "search":
                try:
                    args = json.loads(tc.function.arguments)
                    q_arg = args.get("query", question)
                except Exception:
                    q_arg = question
                return {"action": "search", "query": q_arg}

        content = msg.content or ""
        citations = re.findall(r"\[(stre_[a-zA-Z0-9_]+)\]", content)
        citations = list(dict.fromkeys(citations))

        # Check for abstention
        lower_content = content.lower()
        if "không đủ bằng chứng" in lower_content or "không có thông tin" in lower_content or "không đủ thông tin" in lower_content:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        return {"action": "answer", "answer": content, "citations": citations}


def run_benchmark_agent(split: str = "dev", retriever_type: str = "hybrid_z",
                        max_searches: int = 3, top_k: int = 3, budget: int = 7200):
    cases_path = ROOT / "data" / "agent_cases.json"
    chunks_path = ROOT / "data" / "chunks.json"

    with open(cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    target_cases = [c for c in all_cases if c.get("split") == split]
    print(f"=== Running Agent Benchmark on Split: '{split}' | Retriever: '{retriever_type}' (N = {len(target_cases)} cases) ===")

    print("Initializing Retrievers...")
    dense = DenseRetriever(chunks, qdrant_url="http://localhost:6333", collection_name="collection_strategy_e")
    bm25 = BM25Retriever(chunks)
    
    if retriever_type == "dense":
        active_retriever = dense
    else:
        active_retriever = ZScoreHybridRetriever(dense, bm25, alpha=0.6, candidate_k=15)

    controller = OpenAIController(model="gpt-4o-mini")

    traces = []
    start_all = time.perf_counter()

    for idx, case in enumerate(target_cases, 1):
        cid = case["id"]
        q = case["question"]
        is_ans = case["answerable"]
        print(f"[{idx}/{len(target_cases)}] Query {cid} (answerable={is_ans}): {q[:60]}...")

        t0 = time.perf_counter()
        try:
            res = run_agent(q, controller, active_retriever, filters=case.get("filter"),
                            max_searches=max_searches, top_k=top_k, budget=budget)
            elapsed = time.perf_counter() - t0
            trace_item = {
                "query_id": cid,
                "split": split,
                "answerable": is_ans,
                "question": q,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "elapsed_seconds": round(elapsed, 3)
            }
            num_searches = sum(1 for e in res["events"] if e.get("event") == "search")
            print(f"   -> Status: {res['status']} | Citations: {res['citations']} | Searches: {num_searches} | Time: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"   -> ERROR: {e}")
            trace_item = {
                "query_id": cid,
                "split": split,
                "answerable": is_ans,
                "question": q,
                "status": "error",
                "error_message": str(e),
                "citations": [],
                "events": [],
                "elapsed_seconds": round(elapsed, 3)
            }
        traces.append(trace_item)

    total_time = time.perf_counter() - start_all
    cost_usd = (controller.total_prompt_tokens * 0.10 + controller.total_completion_tokens * 0.50) / 1_000_000

    print("\n" + "=" * 80)
    print(f"Completed Split '{split}' with {retriever_type}: {len(traces)} cases in {total_time:.2f}s (avg {total_time/len(traces):.2f}s/q)")
    print(f"Tokens: {controller.total_prompt_tokens} prompt, {controller.total_completion_tokens} completion")
    print(f"Estimated OpenAI Cost: ${cost_usd:.5f} USD")
    print("=" * 80)

    out_dir = ROOT / "starter" / "results" / f"agent_{split}_{retriever_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    print(f"Saved traces to: {out_dir / 'traces.json'}")

    search_counts = [sum(1 for e in t.get("events", []) if e.get("event") == "search") for t in traces]
    avg_search = round(statistics.mean(search_counts), 2) if search_counts else 0.0
    avg_controller = round(avg_search + 1.0, 2)

    metadata = {
        "split": split,
        "retriever": retriever_type,
        "model": "gpt-4o-mini",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "vector_store": "Qdrant (collection_strategy_e)",
        "context_budget_chars": budget,
        "max_searches": max_searches,
        "top_k": top_k,
        "cases_count": len(traces),
        "answerable_count": sum(1 for t in traces if t.get("answerable")),
        "unanswerable_count": sum(1 for t in traces if not t.get("answerable")),
        "prompt_tokens": controller.total_prompt_tokens,
        "completion_tokens": controller.total_completion_tokens,
        "pricing_per_1m_tokens": {
            "prompt_usd": 0.10,
            "completion_usd": 0.50,
            "reference": "OpenAI official pricing: $0.10/1M input tokens, $0.50/1M output tokens"
        },
        "total_cost_usd": round(cost_usd, 6),
        "total_elapsed_seconds": round(total_time, 2),
        "avg_elapsed_seconds_per_query": round(total_time / len(traces), 2) if traces else 0.0,
        "avg_search_calls_per_query": avg_search,
        "avg_controller_calls_per_query": avg_controller,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_version": "Netflix 2023 Q2 10-Q (88 chunks Strategy E)"
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved run metadata to: {out_dir / 'run_metadata.json'}\n")

    return traces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--retriever", default="hybrid_z", choices=["hybrid_z", "dense"])
    parser.add_argument("--budget", type=int, default=7200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_searches", type=int, default=3)
    args = parser.parse_args()

    run_benchmark_agent(
        split=args.split,
        retriever_type=args.retriever,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget
    )
