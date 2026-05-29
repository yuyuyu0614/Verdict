"""
Regression tests for the noise filtering pipeline.
Run: python -m pytest tests/test_regression.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.l0_engine.noise_engine.pipeline import is_reporter_glue
from src.l0_engine.noise_engine.layer3_syntax import structural_hollowness


class TestReporterGlue:
    def test_meeting_prefix_discard(self):
        assert is_reporter_glue("会议指出") is True
        assert is_reporter_glue("此次会议特别强调") is True

    def test_content_sentence_keep(self):
        assert is_reporter_glue("恒大集团负债2.4万亿元") is False
        assert is_reporter_glue("房地产行业面临调整压力") is False

    def test_source_prefix_discard(self):
        assert is_reporter_glue("据了解") is True
        assert is_reporter_glue("据悉") is True

    def test_person_verb_discard(self):
        assert is_reporter_glue("张三表示") is True


class TestHollowness:
    def test_factual_claim_low_hollow(self):
        score = structural_hollowness("恒大集团负债2.4万亿元", nlp=None)
        assert score < 0.5, f"Expected low hollow, got {score}"

    def test_empty_glue_high_hollow(self):
        score = structural_hollowness("会议指出", nlp=None)
        assert score >= 0.3, f"Expected moderate+ hollow, got {score}"

    def test_numeric_content_low_hollow(self):
        score = structural_hollowness("注册资本为150亿元人民币", nlp=None)
        assert score < 0.5, f"Expected low hollow, got {score}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
