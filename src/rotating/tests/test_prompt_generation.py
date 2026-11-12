"""抽取提示词的公开契约：内容由本体文件生成，改本体即改提示词。"""

from __future__ import annotations

from pathlib import Path

from idr_core.models import BBox, CharSpan, EvidenceSegment
from idr_rotating.ontology import load_ontology
from idr_rotating.prompts import build_extraction_prompt

ALPHA_ONTOLOGY = """
version: 0.1.0
domain: alpha
entity_types:
  alpha_type:
    label: 甲类实体
    description: 甲类描述
    aliases: [甲别名]
    attributes:
      location: 发生部位
relation_types:
  alpha_relation:
    label: 甲关系
    description: 甲关系描述
    domain: [alpha_type]
    range: [alpha_type]
locations:
  alpha_place: 甲部位
instances:
  alpha_type:
    - id: alpha_instance
      label: 甲实例
      aliases: [甲实例别名]
      location: alpha_place
""".strip()

BETA_ONTOLOGY = """
version: 0.2.0
domain: beta
entity_types:
  beta_type:
    label: 乙类实体
    description: 乙类描述
    aliases: [乙别名]
relation_types:
  beta_relation:
    label: 乙关系
    description: 乙关系描述
    domain: [beta_type]
    range: [beta_type]
instances:
  beta_type:
    - id: beta_instance
      label: 乙实例
      aliases: [乙实例别名]
""".strip()


def write_ontology(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body + "\n", encoding="utf-8")
    return path


def evidence_segment() -> EvidenceSegment:
    text = "证据段里提到一个尚未定义的概念。"
    return EvidenceSegment(
        id="ev:doc:1:0-15",
        doc_id="doc",
        page=1,
        bbox=BBox(0.0, 0.0, 10.0, 10.0),
        charspan=CharSpan(0, len(text)),
        text=text,
    )


def test_prompt_changes_when_the_ontology_file_changes(tmp_path: Path) -> None:
    prompt_alpha = build_extraction_prompt(
        load_ontology(write_ontology(tmp_path, "alpha.yaml", ALPHA_ONTOLOGY)),
        (evidence_segment(),),
    )
    prompt_beta = build_extraction_prompt(
        load_ontology(write_ontology(tmp_path, "beta.yaml", BETA_ONTOLOGY)),
        (evidence_segment(),),
    )

    assert prompt_alpha != prompt_beta
    for token in (
        "alpha_type",
        "甲类实体",
        "甲类描述",
        "甲别名",
        "alpha_relation",
        "甲关系",
        "甲关系描述",
        "alpha_instance",
        "甲实例",
        "甲实例别名",
        "alpha_place",
        "甲部位",
    ):
        assert token in prompt_alpha
        assert token not in prompt_beta
    for token in ("beta_type", "乙类实体", "beta_relation", "乙关系", "乙实例"):
        assert token in prompt_beta
        assert token not in prompt_alpha


def test_prompt_lists_every_public_type_and_relation_of_the_real_ontology() -> None:
    ontology = load_ontology()
    segment = evidence_segment()

    prompt = build_extraction_prompt(ontology, (segment,))

    for type_id, definition in ontology.entity_types.items():
        if definition.internal:
            continue
        assert type_id in prompt
        assert definition.label in prompt
    for relation_id, definition in ontology.relation_types.items():
        if definition.internal:
            continue
        assert relation_id in prompt
        assert definition.label in prompt
    assert segment.id in prompt
    assert segment.text in prompt


def test_internal_types_are_not_offered_to_the_model() -> None:
    """证据段这类内部类型由流水线维护，不能出现在抽取提示词里。"""
    ontology = load_ontology()
    prompt = build_extraction_prompt(ontology, (evidence_segment(),))

    # 提示词以「- [类型标识]」列出可抽取类型；内部类型不能出现在清单中。
    assert "[evidence]" not in prompt
    assert "[evidenced_by]" not in prompt
    assert "doc_id" not in prompt

    # 声明是显式的：内部类型仍在本体里，只是不参与模型抽取。
    assert ontology.entity_types["evidence"].internal is True
    assert ontology.relation_types["evidenced_by"].internal is True
    assert ontology.entity_types["fault_mode"].internal is False
    assert ontology.relation_types["indicated_by"].internal is False


def test_prompt_asks_for_location_and_damage_form_with_allowed_values() -> None:
    prompt = build_extraction_prompt(load_ontology(), (evidence_segment(),))

    # 这两个属性在提示词里必须既是「取值清单」也是「输出字段」，否则模型无从填。
    assert '"location"' in prompt
    assert '"damage_form"' in prompt
    assert "outer_race（外圈）" in prompt
    assert "rolling_contact_fatigue（滚动接触疲劳）" in prompt


def test_prompt_asks_for_a_verbatim_supporting_quote_per_assertion() -> None:
    """要求逐字片段，但不得用「摘不出来」当理由让模型丢掉断言。

    早期版本写了「找不到可逐字摘录的片段时不要输出这条断言」，模型据此把
    大量证据充分的断言也一起删了：关系从 180 掉到 130、证据边从 905 掉到
    505、检索命中率从 95.8% 掉到 47.9%。
    """
    prompt = build_extraction_prompt(load_ontology(), (evidence_segment(),))

    assert '"quote"' in prompt
    assert "逐字" in prompt
    # 片段属于所引证据段、不得改写或拼接，否则校验会拒收。
    assert "不要改写" in prompt
    assert "逐字复制" in prompt
    # 摘不出来只是没有锚点，不能因此丢掉断言。
    assert "断言照常" in prompt
    assert "不要输出这条断言" not in prompt
