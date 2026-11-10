"""切分行为：通过 ingest() 门面观察证据段的边界、定位与扫描件隔离。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from idr_core.models import BBox, CharSpan
from idr_core.parsing import ParsedPage
from idr_rotating import DocumentType, IngestDeps, IngestReport, ingest

from fakes import (
    EMPTY_EXTRACTION,
    FakeEmbedder,
    FakeGraphStore,
    FakeLLM,
    FakeParser,
    FakeVectorStore,
    parsed_page,
    stacked_boxes,
)


def write_pdf(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def ingest_pages(
    directory: Path,
    pages: Sequence[ParsedPage],
    *,
    content: bytes = b"%PDF-1.4 fixture",
) -> tuple[IngestReport, FakeGraphStore, FakeLLM]:
    write_pdf(directory, "document.pdf", content)
    graph_store = FakeGraphStore()
    llm = FakeLLM(EMPTY_EXTRACTION)
    report = ingest(
        directory,
        deps=IngestDeps(
            llm=llm,
            parser=FakeParser(pages),
            graph_store=graph_store,
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        ),
    )
    return report, graph_store, llm


def test_patent_paragraph_numbers_split_one_segment_per_paragraph(
    tmp_path: Path,
) -> None:
    front_page = "\n".join(
        [
            "(19)中华人民共和国国家知识产权局",
            "(12)发明专利",
            "(57)摘要",
            "本发明公开了一种滚动轴承故障诊断方法。",
        ]
    )
    description_page = "\n".join(
        [
            "[0001]本发明涉及旋转设备故障诊断技术领域。",
            "[0002]现有技术中，外圈剥落表现为 BPFO 倍频振动。",
            "[0003]本发明的目的在于提高诊断准确率。",
        ]
    )
    pages = [
        parsed_page(1, front_page, line_boxes=stacked_boxes(4)),
        parsed_page(2, description_page, line_boxes=stacked_boxes(3)),
    ]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    document = report.documents[0]
    assert document.document_type is DocumentType.PATENT
    assert document.error is None
    assert document.evidence_segments == 4

    batch = graph_store.batches[0]
    assert [segment.page for segment in batch.evidence] == [1, 2, 2, 2]
    assert [segment.text for segment in batch.evidence] == [
        front_page,
        *description_page.split("\n"),
    ]

    pages_by_number = {page.number: page for page in pages}
    for segment in batch.evidence:
        page = pages_by_number[segment.page]
        assert page.text[segment.charspan.start : segment.charspan.end] == segment.text


def test_paper_is_split_on_heading_hierarchy_without_crossing_sections(
    tmp_path: Path,
) -> None:
    paper_text = "\n".join(
        [
            "1 引言",
            "滚动轴承故障诊断是旋转设备状态监测的核心问题。",
            "故障模式与特征频率的对应关系决定了诊断的准确性。",
            "2 特征频率",
            "外圈故障频率 BPFO 由轴承几何尺寸推导。",
            "2.1 轴承几何",
            "接触角随载荷变化，会影响特征频率的计算。",
            "3 结论",
            "按标题层级切分可保证证据段不跨章节。",
        ]
    )
    pages = [parsed_page(1, paper_text, line_boxes=stacked_boxes(9))]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    document = report.documents[0]
    assert document.document_type is DocumentType.PAPER
    assert document.evidence_segments == 4

    batch = graph_store.batches[0]
    assert [segment.text for segment in batch.evidence] == [
        "\n".join(paper_text.split("\n")[0:3]),
        "\n".join(paper_text.split("\n")[3:5]),
        "\n".join(paper_text.split("\n")[5:7]),
        "\n".join(paper_text.split("\n")[7:9]),
    ]
    assert all(
        segment.text.startswith(("1 引言", "2 特征频率", "2.1 轴承几何", "3 结论"))
        for segment in batch.evidence
    )


def test_scanned_document_is_quarantined_with_reason_before_extraction(
    tmp_path: Path,
) -> None:
    pages = [
        parsed_page(1, ""),
        parsed_page(2, " \n "),
    ]

    report, graph_store, llm = ingest_pages(tmp_path, pages)

    document = report.documents[0]
    assert document.error is None
    assert document.document_type is None
    assert document.evidence_segments == 0
    assert document.quarantine_reason is not None
    assert "扫描件" in document.quarantine_reason
    assert "文字层" in document.quarantine_reason

    assert report.quarantined == 1
    assert report.failed == 0
    assert llm.prompts == []  # 扫描件不调模型
    assert graph_store.batches == []  # 不产出空证据段，也不写图


def test_each_segment_carries_the_bbox_and_charspan_of_its_source_lines(
    tmp_path: Path,
) -> None:
    first_paragraph = "[0001]第一行文字。\n第二行继续描述外圈剥落。"
    second_paragraph = "[0002]另一段文字。"
    page_text = f"{first_paragraph}\n{second_paragraph}"
    first_line_bbox = BBox(72.0, 100.0, 500.0, 112.0)
    second_line_bbox = BBox(72.0, 114.0, 480.0, 126.0)
    third_line_bbox = BBox(72.0, 140.0, 460.0, 152.0)
    pages = [
        parsed_page(
            1,
            page_text,
            line_boxes=[first_line_bbox, second_line_bbox, third_line_bbox],
        )
    ]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    # 没有封面标记，只有段号：文档类型也必须自动判成专利。
    assert report.documents[0].document_type is DocumentType.PATENT
    first, second = graph_store.batches[0].evidence

    assert first.text == first_paragraph
    assert first.charspan == CharSpan(0, len(first_paragraph))
    assert first.bbox == BBox(
        x0=min(first_line_bbox.x0, second_line_bbox.x0),
        y0=min(first_line_bbox.y0, second_line_bbox.y0),
        x1=max(first_line_bbox.x1, second_line_bbox.x1),
        y1=max(first_line_bbox.y1, second_line_bbox.y1),
    )

    assert second.text == second_paragraph
    assert second.charspan == CharSpan(
        len(first_paragraph) + 1, len(page_text)
    )
    assert second.bbox == third_line_bbox
    assert page_text[first.charspan.start : first.charspan.end] == first.text
    assert page_text[second.charspan.start : second.charspan.end] == second.text


def test_patent_without_paragraph_numbers_falls_back_to_paragraph_breaks(
    tmp_path: Path,
) -> None:
    front_page = "\n".join(
        [
            "(19)中华人民共和国国家知识产权局",
            "(12)发明专利",
            "权利要求书",
            "1.一种滚动轴承故障诊断方法，其特征在于采集振动信号。",
        ]
    )
    description_lines = [
        "本发明涉及旋转设备故障诊断技术领域。",
        "现有技术中，外圈剥落表现为 BPFO 倍频振动。",
        "本发明的目的在于提高诊断准确率。",
    ]
    description_text = "\n".join(description_lines)
    pages = [
        parsed_page(1, front_page, line_boxes=stacked_boxes(4)),
        parsed_page(
            2,
            description_text,
            line_boxes=[
                BBox(72.0, 100.0, 472.0, 112.0),
                BBox(72.0, 126.0, 472.0, 138.0),
                BBox(72.0, 152.0, 472.0, 164.0),
            ],
        ),
    ]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    # 靠封面标记判成专利，但正文没有段号，退回按段落间距切。
    assert report.documents[0].document_type is DocumentType.PATENT
    description_segments = [
        segment
        for segment in graph_store.batches[0].evidence
        if segment.page == 2
    ]
    assert [segment.text for segment in description_segments] == description_lines
    for segment, line in zip(
        description_segments, description_lines, strict=True
    ):
        page = pages[1]
        assert page.text[segment.charspan.start : segment.charspan.end] == line


def test_section_heading_is_grouped_with_the_paragraph_it_introduces(
    tmp_path: Path,
) -> None:
    page_text = "\n".join(
        [
            "说明书",
            "[0001]本发明涉及旋转设备故障诊断技术领域。",
            "[0002]现有技术中，外圈剥落表现为 BPFO 倍频振动。",
        ]
    )
    pages = [parsed_page(1, page_text, line_boxes=stacked_boxes(3))]

    _, graph_store, _ = ingest_pages(tmp_path, pages)

    assert [segment.text for segment in graph_store.batches[0].evidence] == [
        "说明书\n[0001]本发明涉及旋转设备故障诊断技术领域。",
        "[0002]现有技术中，外圈剥落表现为 BPFO 倍频振动。",
    ]


def test_patent_section_heading_between_paragraphs_belongs_to_the_next_one(
    tmp_path: Path,
) -> None:
    page_text = "\n".join(
        [
            "[0001]本发明涉及旋转设备故障诊断技术领域。",
            "背景技术",
            "[0002]现有技术中，外圈剥落表现为 BPFO 倍频振动。",
        ]
    )
    pages = [parsed_page(1, page_text, line_boxes=stacked_boxes(3))]

    _, graph_store, _ = ingest_pages(tmp_path, pages)

    assert [segment.text for segment in graph_store.batches[0].evidence] == [
        "[0001]本发明涉及旋转设备故障诊断技术领域。",
        "背景技术\n[0002]现有技术中，外圈剥落表现为 BPFO 倍频振动。",
    ]


def test_paper_section_spanning_pages_is_split_at_the_page_boundary(
    tmp_path: Path,
) -> None:
    page_one = "1 引言\n第一段文字。"
    page_two = "续写引言的内容。\n2 方法\n第二段文字。"
    pages = [
        parsed_page(1, page_one, line_boxes=stacked_boxes(2)),
        parsed_page(2, page_two, line_boxes=stacked_boxes(3)),
    ]

    _, graph_store, _ = ingest_pages(tmp_path, pages)

    assert [
        (segment.page, segment.text)
        for segment in graph_store.batches[0].evidence
    ] == [
        (1, page_one),
        (2, "续写引言的内容。"),
        (2, "2 方法\n第二段文字。"),
    ]


def test_paper_without_headings_still_yields_page_level_segments(
    tmp_path: Path,
) -> None:
    page_one = (
        "滚动轴承故障诊断依赖振动信号分析。\n"
        "外圈剥落会在频谱上留下特征频率。"
    )
    page_two = (
        "内圈剥落同样具有可辨识的特征频率。\n"
        "现场诊断需要结合多种征兆综合判断。"
    )
    pages = [
        parsed_page(1, page_one, line_boxes=stacked_boxes(2)),
        parsed_page(2, page_two, line_boxes=stacked_boxes(2)),
    ]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    assert report.documents[0].document_type is DocumentType.PAPER
    assert [
        (segment.page, segment.text)
        for segment in graph_store.batches[0].evidence
    ] == [(1, page_one), (2, page_two)]


def test_empty_pages_are_skipped_without_losing_page_numbers(tmp_path: Path) -> None:
    page_one = (
        "滚动轴承故障诊断依赖振动信号分析。\n"
        "外圈剥落会在频谱上留下特征频率。"
    )
    page_three = (
        "内圈剥落同样具有可辨识的特征频率。\n"
        "现场诊断需要结合多种征兆综合判断。"
    )
    pages = [
        parsed_page(1, page_one, line_boxes=stacked_boxes(2)),
        parsed_page(2, ""),
        parsed_page(3, page_three, line_boxes=stacked_boxes(2)),
    ]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    assert report.documents[0].error is None
    assert [
        (segment.page, segment.text)
        for segment in graph_store.batches[0].evidence
    ] == [(1, page_one), (3, page_three)]


def test_single_page_patent_is_segmented_into_its_paragraphs(tmp_path: Path) -> None:
    page_text = "[0001]单页专利的第一段。\n[0002]单页专利的第二段。"
    pages = [parsed_page(1, page_text, line_boxes=stacked_boxes(2))]

    report, graph_store, _ = ingest_pages(tmp_path, pages)

    assert report.documents[0].document_type is DocumentType.PATENT
    assert [segment.text for segment in graph_store.batches[0].evidence] == [
        "[0001]单页专利的第一段。",
        "[0002]单页专利的第二段。",
    ]
