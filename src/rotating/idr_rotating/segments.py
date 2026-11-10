"""把解析结果切成证据段。

本票只做最薄的一层：一页一条证据段。论文与专利的结构化切分留给 T3。
"""

from idr_core.models import CharSpan, EvidenceSegment
from idr_core.parsing import ParsedDocument


def build_evidence_segments(
    doc_id: str, parsed: ParsedDocument
) -> tuple[EvidenceSegment, ...]:
    segments: list[EvidenceSegment] = []
    for page in parsed.pages:
        if not page.text.strip():
            continue
        segments.append(
            EvidenceSegment(
                id=f"ev:{doc_id}:{page.number}:0-{len(page.text)}",
                doc_id=doc_id,
                page=page.number,
                bbox=page.bbox,
                charspan=CharSpan(start=0, end=len(page.text)),
                text=page.text,
            )
        )
    return tuple(segments)
