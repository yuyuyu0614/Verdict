# 12-Dimensional Feature Vector Extraction

FEATURE_DIMS = {
    "D1_entity_density": float,       # L0: NER entity count / total chars
    "D2_has_number": int,             # L0: 0/1
    "D3_has_time": int,               # L0: 0/1
    "D4_source_traceable": int,       # L0: 0/1 (has source_info)
    "D5_verb_object_verifiable": float,  # L0: verb+object present
    "D6_sentence_completeness": float,   # L0: relative length score
    "D7_structural_hollowness": float,   # L3: structural hollow score
    "D8_causal_risk": float,             # L3: causal risk score
    "D9_temporal_staleness": float,      # L4: temporal staleness
    "D10_context_quality": float,        # L2: paragraph context quality
    "D11_contradiction_flag": int,       # L4: intra-doc contradiction
    "D12_source_score": float,           # L5: source authority
}

import re


def build_feature_vector(claim: dict, context_quality: float,
                         doc_date: str = None, nlp=None) -> list:
    """
    Extract 12-dimensional feature vector from a claim dict.
    Returns list of 12 float values.
    """
    text = claim.get("claim_text", "")
    entities = claim.get("entity_snapshot", [])

    # D1: entity density (entities per char)
    d1 = len(entities) / max(len(text), 1)

    # D2: has number
    d2 = 1 if re.search(r"\d+", text) else 0

    # D3: has time expression
    d3 = 1 if (claim.get("time_expressions") or
               re.search(r"\d{4}年|\d{1,2}月\d{1,2}日", text)) else 0

    # D4: source traceable
    d4 = 1 if claim.get("source_info") else 0

    # D5: verb-object verifiability (has core_verb + object)
    d5 = 0.5  # default
    if claim.get("core_verb"):
        d5 += 0.3
    if nlp:
        doc = nlp(text)
        has_obj = any(t.dep_ in ("dobj", "obj", "nsubjpass") for t in doc)
        if has_obj:
            d5 += 0.2

    # D6: sentence completeness (relative length, cap at 100 chars)
    d6 = min(len(text) / 100.0, 1.0)

    # D7-D12: from claim dict (set by noise engine layers)
    d7 = claim.get("structural_hollowness", 0.0)
    d8 = claim.get("causal_risk", 0.0)
    d9 = claim.get("temporal_staleness", 0.0)
    d10 = context_quality
    d11 = claim.get("contradiction_flag", 0)
    d12 = claim.get("source_score", 0.5)

    return [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]
