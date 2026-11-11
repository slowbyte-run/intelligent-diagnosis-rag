#!/usr/bin/env python3
"""抓取语料 PDF 并生成可审计的清单。

用法：
    export http_proxy=... https_proxy=...   # 需要时
    uv run --with pdfplumber python corpus/fetch.py

设计约束（见 docs/adr 与 #9 的验收）：
- PDF 落在 data/input/（整个 data/ 已 gitignore，不进仓库）；
- 清单 corpus/manifest.json 进仓库，记录来源、许可证、检索时间、sha256、页数、文字层；
- 已存在的文件不重复下载，除非 --force；
- 不做批量爬取：每个请求之间间隔 REQUEST_DELAY 秒。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
MANIFEST = ROOT / "corpus" / "manifest.json"
REQUEST_DELAY = 6.0

# 每个故障模式至少两篇支撑（见 issue #9 的验收要求）
ARXIV_SEEDS: list[dict[str, str]] = [
    {"id": "2211.09551", "fault_focus": "综述/数据融合"},
    {"id": "2206.14153", "fault_focus": "多故障/旋转机械"},
    {"id": "2111.08185", "fault_focus": "综述/图神经网络"},
    {"id": "2403.17181", "fault_focus": "轴承/信号处理"},
    {"id": "2308.14174", "fault_focus": "齿轮箱"},
    {"id": "2201.13403", "fault_focus": "风电/振动"},
    {"id": "2303.03496", "fault_focus": "齿轮箱/知识驱动"},
    {"id": "1605.04874", "fault_focus": "齿轮箱"},
    {"id": "2305.19569", "fault_focus": "轴承/知识驱动"},
    {"id": "2212.14732", "fault_focus": "不对中/数据集"},
    {"id": "1208.3122", "fault_focus": "转子"},
    {"id": "2512.14210", "fault_focus": "不平衡/轴承"},
]

# 中文专利：件号由检索确定后填入；抓取 Google Patents 页面并本地生成 PDF。
PATENT_SEEDS: list[dict[str, str]] = []


@dataclass
class Entry:
    doc_id: str
    kind: str
    source_url: str
    licence: str = "unknown"
    licence_source: str = "unknown"
    title: str = ""
    fault_focus: str = ""
    retrieved_at: str = ""
    sha256: str = ""
    bytes: int = 0
    pages: int = 0
    text_layer_chars: int = 0
    generated: bool = False
    probe: dict[str, str] = field(default_factory=dict)


def fetch(url: str, *, retries: int = 3, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "idr-corpus/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def arxiv_licence(arxiv_id: str) -> tuple[str, str]:
    """从 abs 页面读许可证；读不到就如实记为 unknown。"""
    try:
        html = fetch(f"https://arxiv.org/abs/{arxiv_id}", timeout=60).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return "unknown", "abs 页面不可达"
    match = re.search(r'class="abs-license"\s*>?\s*<a[^>]+href="([^"]+)"', html)
    href = match.group(1) if match else ""
    if not href and "nonexclusive-distrib" in html:
        href = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
    if not href:
        return "unknown", "abs 页面未找到 license 行"
    if "publicdomain/zero" in href:
        name = "CC0 1.0"
    elif "licenses/by/4.0" in href:
        name = "CC BY 4.0"
    elif "licenses/by-sa" in href:
        name = "CC BY-SA"
    elif "nonexclusive-distrib" in href:
        name = "arXiv perpetual non-exclusive licence"
    else:
        name = href
    return name, f"arxiv abs 页面 {href}"


def inspect_pdf(path: Path) -> tuple[int, int]:
    """返回 (页数, 首页文字层字符数)。"""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        first = pdf.pages[0].extract_text() or "" if pages else ""
    return pages, len(re.sub(r"\s", "", first))


def record(entry: Entry, pdf_path: Path) -> Entry:
    data = pdf_path.read_bytes()
    entry.sha256 = hashlib.sha256(data).hexdigest()
    entry.bytes = len(data)
    entry.pages, entry.text_layer_chars = inspect_pdf(pdf_path)
    entry.retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    return entry


def ingest_arxiv(force: bool) -> list[Entry]:
    entries: list[Entry] = []
    for seed in ARXIV_SEEDS:
        arxiv_id = seed["id"]
        doc_id = f"arxiv:{arxiv_id}"
        target = INPUT_DIR / f"arxiv-{arxiv_id.replace('.', '_')}.pdf"
        existing = {item.doc_id: item for item in load_manifest()}
        if target.exists() and not force and doc_id in existing:
            print(f"跳过下载，重新校验 {target.name}")
            entry = existing[doc_id]
            entry.probe = {"text_layer_ok": "unknown"}
            record(entry, target)
            entry.probe["text_layer_ok"] = (
                "yes" if entry.text_layer_chars > 200 else "no"
            )
            if entry.licence == "unknown":
                time.sleep(REQUEST_DELAY)
                entry.licence, entry.licence_source = arxiv_licence(arxiv_id)
                print(f"  重新取许可证：{entry.licence}")
            entries.append(entry)
            continue
        print(f"下载 {arxiv_id} ...")
        pdf = fetch(f"https://arxiv.org/pdf/{arxiv_id}")
        target.write_bytes(pdf)
        licence, source = arxiv_licence(arxiv_id)
        entry = Entry(
            doc_id=doc_id,
            kind="paper",
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            licence=licence,
            licence_source=source,
            fault_focus=seed.get("fault_focus", ""),
            probe={"text_layer_ok": "unknown"},
        )
        record(entry, target)
        entry.probe["text_layer_ok"] = (
            "yes" if entry.text_layer_chars > 200 else "no"
        )
        entries.append(entry)
        time.sleep(REQUEST_DELAY)
    return entries


def load_manifest() -> list[Entry]:
    if not MANIFEST.exists():
        return []
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [Entry(**item) for item in raw.get("documents", [])]


def write_manifest(entries: list[Entry]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "note": "PDF 不入库；本清单只记录来源与校验信息，用于复现与审计。",
        "documents": [entry.__dict__ for entry in entries],
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="重新下载已存在的文件")
    parser.add_argument("--only", choices=["arxiv", "patents"], default=None)
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()

    if args.only in (None, "arxiv"):
        entries = ingest_arxiv(args.force)
    if args.only in (None, "patents"):
        # 专利抓取在 fetch_patents.py 中实现；本脚本只保留入口占位。
        print("专利抓取见 corpus/fetch_patents.py")

    write_manifest(entries)
    ok = sum(1 for entry in entries if entry.probe.get("text_layer_ok") == "yes")
    print(f"\n清单已写入 {MANIFEST.relative_to(ROOT)}：{len(entries)} 篇，{ok} 篇有文字层")
    return 0


if __name__ == "__main__":
    sys.exit(main())
