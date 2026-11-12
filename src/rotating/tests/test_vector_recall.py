"""向量兜底的候选融合：两路召回按名次合成一个候选集。

向量距离与词面命中数的尺度不可比，融合只能用名次（RRF）；这组测试只锁
「名次怎么变成候选集」，不重复测召回与重排。
"""

from __future__ import annotations

from idr_rotating.vector_recall import fuse_rankings


def test_without_lexical_hits_the_fused_list_is_the_vector_order() -> None:
    assert fuse_rankings(("a", "b", "c"), (), limit=3) == ("a", "b", "c")


def test_a_candidate_hit_by_both_channels_moves_up() -> None:
    # b 在向量里第 2、词面里第 1，分数 1/62+1/61，高于只在向量里第 1 的 a。
    assert fuse_rankings(("a", "b"), ("b",), limit=3) == ("b", "a")


def test_equal_scores_are_broken_by_evidence_id_for_reproducibility() -> None:
    assert fuse_rankings(("z",), ("a",), limit=2) == ("a", "z")


def test_the_fused_set_is_truncated_to_the_rerank_budget() -> None:
    assert fuse_rankings(("a", "b"), ("c", "d"), limit=2) == ("a", "c")


def test_a_candidate_present_in_both_channels_appears_once() -> None:
    fused = fuse_rankings(("a", "b"), ("b", "c"), limit=10)

    assert fused == ("b", "a", "c")
    assert len(fused) == len(set(fused))
