"""
main.py 閳?Verdict FastAPI Service
POST /evaluate  -> L0 atomize -> L2 search+score -> L3 fuse
POST /atomize   -> L0 only (fast path, no search)
GET  /health    -> service status
"""

import os, sys, time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Ensure local packages are importable
_pkg_root = os.path.join(os.path.dirname(__file__), "..", "packages")
for _pkg in ["fact-atomizer", "content-quality-judge", "decision-fusion"]:
    _path = os.path.join(_pkg_root, _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sys as _sys, os as _os
_api_root = _os.path.dirname(__file__)
if _api_root not in _sys.path:
    _sys.path.insert(0, _api_root)
_proj_root = _os.path.join(_os.path.dirname(__file__), "..")
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)
from src.l0_engine import RuleEngine
from content_quality_judge import ContentQualityJudge
from decision_fusion import fuse
from config import load as load_config
from local_search import init_kb, search_passthrough
from verification import run_verification
from authority import get_authority_bonus
from timeline_client import search as timeline_search, _save_key, get_key_info, download_nodes, download_all, send_feedback, health_check as timeline_health
from api.timeline_atoms import batch_search_cached, compute_l2_bonus
# --- Config ---
cfg = load_config()

# --- Lifespan: init modules at startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # L0: RuleEngine (deterministic, zero LLM) - no API key needed
    init_kb()  # Initialize local knowledge base

    # Sync rules from TimeLine rule center
    from src.rules_sync import sync_rules
    sync_rules()

    app.state.judge = ContentQualityJudge(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
    )
    yield

app = FastAPI(title="Verdict", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Models ---

class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=cfg["max_text_length"], description="Text to evaluate")


class ClaimResult(BaseModel):
    claim_id: int
    claim_text: str
    claim_type: str = ""
    entities: list[str] = []
    l2_score: float = 0.0
    l2_label: str = ""
    l3_score: float = 0.0
    l3_label: str = ""
    l3_confidence: str = ""
    cross_validated: bool = False


class EvaluateResponse(BaseModel):
    claims: list[ClaimResult]
    claim_count: int
    overall_score: float = Field(description="0.0-1.0 aggregate score across all claims")
    verdict: str = Field(description="verified / reliable / speculation / low_quality / unconfirmed")
    verification: Optional[dict] = Field(default=None, description="V8.0 verification gate result")


class TimelineKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=8, description="TimeLine API Key")
    label: str = Field(default="", description="Key label for identification")


class DownloadRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000, description="Nodes per batch")
    offset: int = Field(default=0, ge=0)
    max_total: int = Field(default=500, ge=1, le=10000)
    category: Optional[str] = Field(default=None)
    min_credibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    claim_id: int
    claim_text: str
    l3_score: float
    l3_label: str
    sources: Optional[list[str]] = None
    verification: Optional[dict] = None


# --- Verdict logic ---

def _compute_verdict(claims: list[dict]) -> tuple[float, str]:
    if not claims:
        return 0.0, "unconfirmed"
    scores = [c.get("l3_score", 0) for c in claims]
    avg = sum(scores) / len(scores) if scores else 0.0
    if avg >= 0.8:
        verdict = "verified"
    elif avg >= 0.6:
        verdict = "reliable"
    elif avg >= 0.35:
        verdict = "speculation"
    elif avg >= 0.15:
        verdict = "low_quality"
    else:
        verdict = "unconfirmed"
    return round(avg, 3), verdict


# --- Endpoints ---

_DEMO_PATH = _os.path.join(_os.path.dirname(__file__), "..", "demo_ui.html")

PRESETS = [
    {
        "label": "Evergrande auction",
        "text": "5月3日，恒大地产集团有限公司位于广州市天河区的一块商业用地将于5月5日公开拍卖，起始价为1949万元。恒大地产集团有限公司成立于1996年，注册资本为150亿元人民币。",
    },
    {
        "label": "Health claim",
        "text": "每天饮用苹果醋可以有效降低血糖。有健身博主推测，高强度间歇训练的效果可能是传统有氧运动的2倍以上。",
    },
    {
        "label": "Policy statement",
        "text": "双方一致认为，两国经贸合作前景广阔，双方决定进一步扩大投资规模。2024年双边贸易额突破500亿美元。",
    },
]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": cfg["model"],
        "search_enabled": cfg["search_enabled"],
    }


@app.get("/presets")
def presets():
    return PRESETS


@app.get("/")
def index():
    return FileResponse(_DEMO_PATH)


@app.post("/atomize", response_model=dict)
def atomize(req: EvaluateRequest):
    """L0 only: decompose text into atomic claims without scoring."""
    try:
        claims = RuleEngine.run(req.text)
    except Exception as e:
        raise HTTPException(500, f"L0 atomize failed: {e}")
    return {"claims": claims, "claim_count": len(claims)}



@app.post("/evaluate", response_model=EvaluateResponse, response_model_exclude_none=False)
def evaluate(req: EvaluateRequest):
    """Full pipeline: L0 -> search -> L2 -> L3."""

    # [V8.0] Verification gate: check document references
    v_result = run_verification(req.text)
    if v_result["verdict"] == "all_failed":
        resp = EvaluateResponse(
            claims=[], claim_count=0, overall_score=0.0,
            verdict="verification_failed",
            verification=v_result,
        )

    # L0
    try:
        raw_claims = RuleEngine.run(req.text)
    except Exception as e:
        raise HTTPException(500, f"L0 atomize failed: {e}")

    if not raw_claims:
        return EvaluateResponse(claims=[], claim_count=0, overall_score=0.0, verdict="unconfirmed")

    results = []
    for i, c in enumerate(raw_claims):
        if "error" in c:
            results.append(dict(
                claim_id=i+1, claim_text=c.get("claim_text","?"), claim_type="error",
                entities=[], l2_score=0, l2_label="error", l3_score=0, l3_label="error",
                l3_confidence="unknown", cross_validated=False,
            ))
            continue

        claim_text = c["claim_text"]
        ct = c.get("claim_type", "?")
        ents = c.get("entities_mentioned", [])

        # Search
        url_a = title_a = snippet_a = ""
        url_b = title_b = snippet_b = ""
        has_a = has_b = False

        if cfg["search_enabled"]:
            try:
                if cfg.get("search_mode") == "local":
                    hits = search_passthrough(claim_text, top_k=2)
                else:
                    hits = timeline_search(claim_text, limit=2)
                if len(hits) >= 1:
                    url_a = hits[0].get("href", "")
                    title_a = hits[0].get("title", "")
                    snippet_a = (hits[0].get("body", "") or "")[:200]
                    has_a = True
                if len(hits) >= 2:
                    url_b = hits[1].get("href", "")
                    title_b = hits[1].get("title", "")
                    snippet_b = (hits[1].get("body", "") or "")[:200]
                    has_b = True
            except Exception:
                pass

        # L2
        score_a = score_b = 0.0
        label_a = label_b = "no_result"
        if has_a:
            try:
                r = app.state.judge.evaluate(claim_text, title_a or claim_text, snippet_a or claim_text, url_a.split("/")[-1] if url_a else "unknown")
                score_a = r.get("content_quality_score", 0.0)
                label_a = r.get("domain_type", "?")
            except Exception:
                pass
        if has_b:
            time.sleep(0.3)
            try:
                r = app.state.judge.evaluate(claim_text, title_b or claim_text, snippet_b or claim_text, url_b.split("/")[-1] if url_b else "unknown")
                score_b = r.get("content_quality_score", 0.0)
                label_b = r.get("domain_type", "?")
            except Exception:
                pass
        
        # Fallback: no search results -> evaluate claim self-signals
        if not has_a and not has_b:
            try:
                r = app.state.judge.evaluate(
                    claim_text,
                    title="(self-evaluation, no search result)",
                    snippet=claim_text,
                    domain="self-signal"
                )
                score_a = r.get("content_quality_score", 0.5)
                label_a = r.get("domain_type", "self_signal")
                has_a = True
            except Exception:
                score_a = 0.5
                label_a = "self_signal_fallback"
                has_a = True
        

        # L3
        # L3: compute authority bonus from domain whitelist
        auth_a = get_authority_bonus(url_a) if url_a else 0.0
        auth_b = get_authority_bonus(url_b) if url_b else 0.0

        fused = fuse(
            source_a_quality=score_a,
            source_b_quality=score_b,
            url_a=url_a, url_b=url_b,
            has_source_a=has_a, has_source_b=has_b,
            consistent=(abs(score_a - score_b) < 0.4) if (has_a and has_b) else False,
            authority_bonus_a=auth_a,
            authority_bonus_b=auth_b,
        )

        results.append(dict(
            claim_id=i+1,
            claim_text=claim_text,
            claim_type=ct,
            entities=ents[:5],
            l2_score=max(score_a, score_b),
            l2_label=label_a if score_a >= score_b else label_b,
            l3_score=fused["final_score"],
            l3_label=fused["label"],
            l3_confidence=fused["confidence"],
            cross_validated=fused["cross_validation"]["bonus"] >= 1.3,
        ))

        time.sleep(0.5)

    overall, verdict = _compute_verdict(results)
    return EvaluateResponse(
        claims=[ClaimResult(**r) for r in results],
        claim_count=len(results),
        overall_score=overall,
        verdict=verdict,
        verification=v_result,
    )


# ========== TimeLine Integration Endpoints ==========

@app.get("/timeline/key")
def timeline_key_info():
    """Get current TimeLine API key info (prefix only, no full key exposed)."""
    return get_key_info()


@app.post("/timeline/key")
def timeline_key_set(req: TimelineKeyRequest):
    """Save TimeLine API key for Verdict to use."""
    _save_key(req.api_key, req.label)
    info = get_key_info()
    return {"status": "saved", "key_prefix": info["key_prefix"], "label": req.label}


@app.get("/timeline/health")
def timeline_health_check():
    """Check TimeLine service reachability."""
    return timeline_health()


@app.post("/timeline/download")
def timeline_download(req: DownloadRequest = None):
    """
    Download nodes from TimeLine.
    If max_total > limit, paginates automatically.
    """
    if req is None:
        req = DownloadRequest()
    
    if req.max_total <= req.limit:
        result = download_nodes(
            limit=req.max_total,
            offset=req.offset,
            category=req.category,
            min_credibility=req.min_credibility,
        )
    else:
        result = download_all(
            batch_size=req.limit,
            max_total=req.max_total,
            category=req.category,
            min_credibility=req.min_credibility,
        )
    
    return {
        "status": "ok",
        "downloaded": result["total_downloaded"] if "total_downloaded" in result else result["downloaded"],
        "total_available": result.get("total", result.get("total_downloaded")),
        "has_more": result.get("has_more", False),
        "sample": result["nodes"][:3] if result.get("nodes") else [],
    }


@app.post("/timeline/feedback")
def timeline_feedback(req: FeedbackRequest):
    """
    Send Verdict evaluation result back to TimeLine.
    This closes the loop: TimeLine raw data -> Verdict evaluation -> feedback to TimeLine.
    """
    try:
        result = send_feedback(
            claim_id=req.claim_id,
            claim_text=req.claim_text,
            l3_score=req.l3_score,
            l3_label=req.l3_label,
            sources=req.sources,
            verification_result=req.verification,
        )
        return {"status": "sent", "response": result}
    except Exception as e:
        raise HTTPException(502, f"TimeLine feedback failed: {e}")




# ========== V10: Zero-LLM Noise-Aware Evaluate ==========

from src.l0_engine.atomizer import atomize as l0_atomize
from src.l0_engine.noise_engine.pipeline import filter_noise


@app.post("/evaluate_v2")
async def evaluate_v2(req: EvaluateRequest):
    """
    L0 atomize -> L0.5 noise filter -> L2 TimeLine cross-validation -> results.
    TimeLine L2 is a bonus layer; degrades gracefully if unreachable.
    """
    # L0: Atomize
    claims = l0_atomize(req.text)

    # L0.5: Noise filter
    clean_claims = filter_noise(claims, req.text)

    # L2: no-op (TimeLine integration removed for open-source release)
    for cc in clean_claims:
        cc.update({
            "cross_validation_count": 0,
            "cross_validation_score": 0.0,
            "external_sources": [],
        })

    # Build results
    results = []
    all_texts = {c["claim_text"]: c for c in clean_claims}
    for i, c in enumerate(claims):
        text = c["claim_text"]
        if text in all_texts:
            cc = all_texts[text]
            ns = cc.get("noise_score", 0)
            review = cc.get("needs_review", False)
            if ns >= 0.65:
                label = "noise_discard"
            elif review:
                label = "needs_review"
            else:
                label = "clean"
            results.append({
                "claim_id": c["claim_id"],
                "claim_text": text,
                "noise_score": round(ns, 4),
                "label": label,
                "structural_hollowness": round(cc.get("structural_hollowness", 0), 4),
                "causal_risk": round(cc.get("causal_risk", 0), 4),
                "causal_links": c.get("causal_links", []),
                "is_speculation": c.get("is_speculation", False),
                "source_info": c.get("source_info"),
                "needs_review": review,
                "cross_validation_count": cc.get("cross_validation_count", 0),
                "cross_validation_score": cc.get("cross_validation_score", 0.0),
                "external_sources": cc.get("external_sources", []),
            })
        else:
            results.append({
                "claim_id": c["claim_id"],
                "claim_text": text,
                "noise_score": 1.0,
                "label": "discarded_l1",
                "structural_hollowness": 0,
                "causal_risk": 0,
                "causal_links": c.get("causal_links", []),
                "is_speculation": c.get("is_speculation", False),
                "source_info": c.get("source_info"),
                "needs_review": False,
                "cross_validation_count": 0,
                "cross_validation_score": 0.0,
                "external_sources": [],
            })

    total = len(results)
    noise_count = sum(1 for r in results if r["label"] in ("noise_discard", "discarded_l1"))
    review_count = sum(1 for r in results if r["needs_review"])
    clean_count = total - noise_count

    return {
        "total_claims": total,
        "clean_claims": clean_count,
        "noise_claims": noise_count,
        "review_claims": review_count,
        "noise_rate": round(noise_count / max(total, 1), 4),
        "claims": results,
        "verdict": "noise_aware" if noise_count == 0 else ("contains_noise" if noise_count <= total * 0.3 else "high_noise"),
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Verdict API starting on http://{cfg['host']}:{cfg['port']}")
    print(f"Model: {cfg['model']} | Search: {cfg['search_enabled']}")
    uvicorn.run(app, host=cfg["host"], port=cfg["port"])



