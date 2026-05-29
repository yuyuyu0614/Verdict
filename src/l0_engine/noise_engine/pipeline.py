# filter_noise() — Real-time noise filtering pipeline
# Orchestrates all five layers + fusion

import sqlite3
import json
import os
from datetime import datetime

from .layer1_rules import check_ad_filter
from .layer2_info import compute_context_quality
from .layer3_syntax import structural_hollowness, causal_risk
from .layer4_logic import temporal_staleness, contradiction_check
from .layer5_source import source_score
from .feature_vector import build_feature_vector
from .fusion import predict_noise_score, classify_noise

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "data", "noise_log.db")



# Reporter glue patterns: pure pattern match, no hollow dependency
import re as _re

REPORTER_GLUE_PATTERNS = [
    _re.compile(r'^(此次)?会议(指出|强调|认为|表示|还指出|特别强调)'),
    _re.compile(r'^.{2,6}(表示|认为|强调|指出|称|透露|介绍)'),
    _re.compile(r'^(据了解|据悉|据报道|据介绍)'),
    _re.compile(r'^(通知称|公告称|声明称)'),
]


def is_reporter_glue(claim_text):
    for pat in REPORTER_GLUE_PATTERNS:
        if pat.match(claim_text):
            return True
    return False

def _init_db():
    """Ensure noise_log.db exists with schema."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS noise_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_text TEXT,
            noise_score REAL,
            noise_level TEXT,
            discarded INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def _log_noise(claim_text: str, noise_score: float,
               noise_level: str, discarded: bool):
    """Log noise claim to sqlite for flywheel analysis."""
    try:
        _init_db()
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "INSERT INTO noise_log (claim_text, noise_score, noise_level, discarded, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (claim_text, noise_score, noise_level, int(discarded),
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # non-critical


def filter_noise(claims: list, original_text: str,
                 source_url: str = None, source_name: str = None,
                 doc_date: str = None, nlp=None) -> list:
    """
    Hybrid noise filtering: L1 deterministic + LR for borderline.
    spaCy loaded once (not per-claim).
    """
    from .layer3_syntax import structural_hollowness, causal_risk
    from .layer2_info import compute_context_quality
    from .feature_vector import build_feature_vector

    context_quality = compute_context_quality(original_text)

    # Load spaCy once (outside loop)
    spacy_nlp = None
    try:
        import spacy
        spacy_nlp = spacy.load('zh_core_web_sm')
    except Exception:
        pass

    clean_claims = []

    for i, claim in enumerate(claims):
        text = claim.get("claim_text", "")

        # L1: Ad/spam filter
        if check_ad_filter(text):
            _log_noise(text, 1.0, "L1_ad", discarded=True)
            continue

        # Fragment filter
        if len(text) < 4 and not claim.get("has_number"):
            _log_noise(text, 0.8, "L0_fragment", discarded=True)
            continue

        # Reporter glue: pure pattern match before hollow check
        if is_reporter_glue(text):
            _log_noise(text, 0.9, "L2_reporter_glue", discarded=True)
            continue

        # L2: Hollow check
        sh = structural_hollowness(text, nlp=None)
        claim["structural_hollowness"] = sh
        if sh >= 0.7:
            _log_noise(text, float(sh), "L2_extreme_hollow", discarded=True)
            continue

        # L3: Causal risk
        neighbor = claims[i - 1] if i > 0 else None
        neighbors = [neighbor] if neighbor else None
        try:
            cr = causal_risk(text, spacy_nlp, neighbors) if spacy_nlp else causal_risk(text, None, neighbors)
        except Exception:
            cr = causal_risk(text, None, neighbors)
        claim["causal_risk"] = cr

        # L4: Temporal staleness
        claim["temporal_staleness"] = temporal_staleness(text, doc_date)
        claim["contradiction_flag"] = 0

        # L5: Source score
        claim["source_score"] = source_score(source_url, source_name)

        # LR scoring
        vec = build_feature_vector(claim, context_quality, doc_date, nlp)
        noise_score = predict_noise_score(vec)
        claim["noise_score"] = round(noise_score, 4)

        decision = classify_noise(noise_score)

        if decision == "discard":
            _log_noise(text, noise_score, "LR_fusion", discarded=True)
            continue
        elif decision == "review":
            claim["needs_review"] = True
            _log_noise(text, noise_score, "review", discarded=False)
        else:
            _log_noise(text, noise_score, "pass", discarded=False)

        clean_claims.append(claim)

    return clean_claims


if __name__ == "__main__":
    # Quick smoke test
    claims = [
        {"claim_text": "恒大集团负债2.4万亿元", "paragraph_id": 0,
         "entity_snapshot": ["恒大集团"], "time_expressions": [],
         "core_verb": "负债", "source_info": None},
        {"claim_text": "扫码加群领取牛股", "paragraph_id": 0,
         "entity_snapshot": [], "time_expressions": [],
         "core_verb": None, "source_info": None},
    ]
    result = filter_noise(claims, "恒大集团负债2.4万亿元，扫码加群领取牛股")
    print(f"Input: {len(claims)}, Output: {len(result)}")
    for c in result:
        print(f"  [{c.get('noise_score', '?')}] {c['claim_text']}")
