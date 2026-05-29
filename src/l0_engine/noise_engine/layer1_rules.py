# L1: Ad/Spam quick filter
# Rule-based keyword matching with exemption logic

import os
import re
import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "layer1_keywords.yaml")

with open(_YAML_PATH, encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)


def check_ad_filter(claim_text: str) -> bool:
    """
    Returns True if claim should be discarded as ad/spam.
    Exemption: if an exemption keyword appears in the same sentence,
    the claim is NOT discarded.
    """
    for pattern in _CONFIG.get("patterns", []):
        for kw in pattern.get("keywords", []):
            if kw in claim_text:
                for exempt in pattern.get("exemption", []):
                    if exempt in claim_text:
                        return False
                return True
    return False


def export_candidates(candidates: list, path: str):
    """Export candidate keywords to YAML for human review."""
    result_lines = ["# L1 Candidate Keywords (pending review)", "candidates:"]
    for ngram, freq in candidates:
        result_lines.append("  - ngram: " + repr(ngram))
        result_lines.append("    frequency: " + str(freq))
    with open(path, "w", encoding="utf-8") as f:
        f.write(chr(10).join(result_lines))