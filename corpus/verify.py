#!/usr/bin/env python3
"""验收检查器：随机抽断言，沿证据边回溯到原文，核对页码与字符区间。

对应 issue #9 的验收判据：
    20 篇文献入库后，随机抽 10 条断言，每条都能沿 evidenced_by 追回原文段落，且页码正确。

做法（不信任图谱自己记的页码，回到源 PDF 重算）：
1. 从图谱里随机取 N 条带证据的**概念间断言**；
2. 取该断言的证据段 ID，读出证据节点的页码与字符区间；
3. 打开对应源 PDF 的第 page 页，取出该页文字；
4. 断言 page_text[start:end] 与证据段文本完全一致。

用法：
    uv run --with pdfplumber python corpus/verify.py [--sample 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"


def load_pdf_index() -> dict[str, Path]:
    """内容哈希前缀 → PDF 路径。doc_id 由内容哈希推导，不依赖文件名。"""
    index: dict[str, Path] = {}
    for path in sorted(INPUT_DIR.glob("*.pdf")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        index[digest] = path
    return index


def page_text(path: Path, number: int) -> str:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        if number < 1 or number > len(pdf.pages):
            raise IndexError(f"{path.name} 只有 {len(pdf.pages)} 页，证据段声称第 {number} 页")
        return pdf.pages[number - 1].extract_text() or ""


def parse_charspan(value: str) -> tuple[int, int]:
    start, _, end = value.partition("-")
    return int(start), int(end)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    index = load_pdf_index()

    with driver.session() as session:
        candidates = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) <> 'EVIDENCED_BY'
              AND r.evidence_ids IS NOT NULL
              AND size(r.evidence_ids) > 0
            RETURN a.label AS source_label, type(r) AS relation,
                   b.label AS target_label, r.evidence_ids AS evidence_ids
            """
        ).data()

        if not candidates:
            print("图谱里没有任何带证据的断言，无法验收", file=sys.stderr)
            return 1

        random.seed(args.seed)
        sample = random.sample(candidates, min(args.sample, len(candidates)))

        results = []
        for item in sample:
            evidence_id = item["evidence_ids"][0]
            record = session.run(
                """
                MATCH (e:Evidence {id: $id})
                RETURN e.doc_id AS doc_id, e.page AS page,
                       e.charspan AS charspan, e.text AS text
                """,
                id=evidence_id,
            ).single()

            assertion = f"{item['source_label']} -[{item['relation']}]-> {item['target_label']}"
            if record is None:
                results.append({"assertion": assertion, "evidence_id": evidence_id,
                                "ok": False, "reason": "证据节点不存在"})
                continue

            pdf_path = index.get(record["doc_id"])
            if pdf_path is None:
                results.append({"assertion": assertion, "evidence_id": evidence_id,
                                "ok": False, "reason": "找不到对应源 PDF"})
                continue

            try:
                text = page_text(pdf_path, int(record["page"]))
            except IndexError as error:
                results.append({"assertion": assertion, "evidence_id": evidence_id,
                                "ok": False, "reason": str(error)})
                continue

            start, end = parse_charspan(record["charspan"])
            sliced = text[start:end]
            ok = sliced == record["text"]
            results.append({
                "assertion": assertion,
                "evidence_id": evidence_id,
                "pdf": pdf_path.name,
                "page": int(record["page"]),
                "ok": ok,
                "reason": "一致" if ok else "页码或字符区间与源 PDF 不符",
                "expected": record["text"][:60],
                "actual": sliced[:60],
            })

    driver.close()

    passed = sum(1 for item in results if item["ok"])
    report = {
        "sample_size": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "candidates": len(candidates),
        "results": results,
    }
    out = ROOT / "corpus" / "acceptance-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in results:
        mark = "✓" if item["ok"] else "✗"
        detail = item.get("pdf", "")
        page = item.get("page", "")
        print(f"{mark} {item['assertion'][:58]} | {detail} p{page} | {item['reason']}")

    print(f"\n抽样 {len(results)} 条，通过 {passed} 条 → {out.relative_to(ROOT)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
