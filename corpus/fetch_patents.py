#!/usr/bin/env python3
"""抓取中文专利全文并生成本地 PDF。

为什么需要「生成」而不是「下载」：CNIPA 没有可编程接口（反爬 412），公开渠道
只有 Google Patents 的网页全文。若直接存 HTML，就绕过了整条 PDF 解析链路，
验收会变成假验收；因此把正文按段落还原成带 [0001] 标记的 PDF，让证据段、
页码与 bbox 都有确定含义。

合规边界：
- 只取 robots.txt 明确 Allow 的 /patent/ 页面，逐页、带延时，不做批量检索；
- 清单记录来源 URL 与「无内容许可声明」这一事实，PDF 本身不入库。
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
import time
from pathlib import Path

from fetch import Entry, INPUT_DIR, MANIFEST, REQUEST_DELAY, fetch, inspect_pdf, load_manifest, write_manifest

# 八件覆盖五种故障模式的中文专利（件号由公开检索确定）
PATENT_SEEDS: list[dict[str, str]] = [
    {"num": "CN119442026A", "fault_focus": "轴承/故障诊断方法"},
    {"num": "CN117969094A", "fault_focus": "轴承/故障诊断"},
    {"num": "CN203324012U", "fault_focus": "轴承/检测装置"},
    {"num": "CN105352726A", "fault_focus": "齿轮/点蚀与断齿"},
    {"num": "CN106596116A", "fault_focus": "齿轮/振动诊断"},
    {"num": "CN115389190B", "fault_focus": "转子不平衡"},
    {"num": "CN101929917B", "fault_focus": "转子/矢面积法"},
    {"num": "CN104880291A", "fault_focus": "基础松动/轴向振动"},
]

FONT = "STSong-Light"


def clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def section(html: str, name: str) -> str:
    match = re.search(
        rf'<section[^>]*itemprop="{name}"[^>]*>(.*?)</section>', html, re.S
    )
    return match.group(1) if match else ""


def numbered_paragraphs(fragment: str) -> list[tuple[str, str]]:
    pairs = re.findall(r'<div[^>]*num="(\d+)"[^>]*>(.*?)</div>', fragment, re.S)
    result: list[tuple[str, str]] = []
    for number, body in pairs:
        text = clean(body)
        if text:
            result.append((number, text))
    return result


def claim_texts(fragment: str) -> list[str]:
    blocks = re.findall(r'<div[^>]*class="claim[^"]*"[^>]*>(.*?)</div>', fragment, re.S)
    cleaned = [clean(block) for block in blocks]
    return [text for text in cleaned if text]


def patent_title(html: str) -> str:
    match = re.search(r'<meta name="DC.title" content="([^"]*)"', html)
    return html_lib.unescape(match.group(1)).strip() if match else ""


def build_pdf(html: str, target: Path) -> int:
    """把专利正文写成 PDF，返回写入的段落数。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    body = ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=16)
    heading = ParagraphStyle("heading", fontName=FONT, fontSize=13, leading=20)

    title = patent_title(html)
    abstract = clean(section(html, "abstract"))
    claims = claim_texts(section(html, "claims"))
    paragraphs = numbered_paragraphs(section(html, "description"))

    if not paragraphs:
        raise RuntimeError("页面里没有找到带段号的说明书正文")

    story = [Paragraph(html_lib.escape(title or "专利文献"), heading)]
    if abstract:
        story.append(Paragraph("摘要", heading))
        story.append(Paragraph(html_lib.escape(abstract), body))
    if claims:
        story.append(Paragraph("权利要求书", heading))
        for index, text in enumerate(claims, start=1):
            story.append(Paragraph(html_lib.escape(f"{index}. {text}"), body))
    story.append(Paragraph("说明书", heading))
    for number, text in paragraphs:
        story.append(Paragraph(html_lib.escape(f"[{number}] {text}"), body))

    SimpleDocTemplate(
        str(target), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    ).build(story)
    return len(paragraphs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = {entry.doc_id: entry for entry in load_manifest()}

    for seed in PATENT_SEEDS:
        number = seed["num"]
        doc_id = f"patent:{number}"
        target = INPUT_DIR / f"patent-{number}.pdf"
        if target.exists() and not args.force and doc_id in entries:
            entry = entries[doc_id]
            entry.probe = {"text_layer_ok": "unknown"}
            entry.pages, entry.text_layer_chars = inspect_pdf(target)
            entry.probe["text_layer_ok"] = "yes" if entry.text_layer_chars > 200 else "no"
            print(f"跳过下载，重新校验 {target.name}")
            continue

        url = f"https://patents.google.com/patent/{number}/zh"
        print(f"抓取 {number} ...")
        try:
            html = fetch(url).decode("utf-8", "ignore")
            paragraphs = build_pdf(html, target)
        except Exception as error:  # noqa: BLE001 - 单件失败不拖垮整批
            print(f"  跳过 {number}：{type(error).__name__}: {error}")
            target.unlink(missing_ok=True)
            time.sleep(REQUEST_DELAY)
            continue

        entry = Entry(
            doc_id=doc_id,
            kind="patent",
            source_url=url,
            licence="未声明（Google Patents 公开的专利文献）",
            licence_source="Google Patents 页面未提供内容许可声明",
            title=patent_title(html),
            fault_focus=seed["fault_focus"],
            generated=True,
            probe={"paragraphs": str(paragraphs), "text_layer_ok": "unknown"},
        )
        entries[doc_id] = entry
        entry.pages, entry.text_layer_chars = inspect_pdf(target)
        entry.probe["text_layer_ok"] = "yes" if entry.text_layer_chars > 200 else "no"
        print(f"  生成 {entry.pages} 页，{paragraphs} 段")
        time.sleep(REQUEST_DELAY)

    write_manifest(list(entries.values()))
    ok = sum(1 for entry in entries.values() if entry.probe.get("text_layer_ok") == "yes")
    print(f"\n清单共 {len(entries)} 篇，{ok} 篇有文字层 → {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
