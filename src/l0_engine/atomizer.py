# L0 Atomic Claim Atomizer
# Four sub-layers: preprocess -> coarse split -> fine split -> metadata binding
# Revision 2026-05-28:
#   - Causal protection: split at causal conjunctions, tag causal_links
#   - SimHash dedup: in layer 4 (metadata binding), after claims assembled

import re
import hashlib

# spaCy lazy load (hybrid: jieba for hollowness, spaCy for NER/causal)
try:
    import spacy
    _NLP = spacy.load('zh_core_web_sm')
except Exception:
    _NLP = None


# ============================================================
# Layer 1: Preprocessing
# ============================================================

def preprocess(text: str, is_html: bool = False) -> tuple:
    """
    Returns: (clean_text, meta)
    meta = {'quote_ranges': [...], 'noise_paragraphs': [...], 'paragraph_map': {...}}
    """
    meta = {"quote_ranges": [], "noise_paragraphs": [], "paragraph_map": {}}

    # Step 1: HTML tag removal
    if is_html:
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n", text)
        text = re.sub(r"<br[^>]*>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)

    # Step 2: Quote normalization
    quote_map = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
    }
    for fancy, plain in quote_map.items():
        text = text.replace(fancy, plain)

    # Record quote ranges
    for m in re.finditer(r'"[^"]*"', text):
        meta["quote_ranges"].append((m.start(), m.end()))

    # Step 3: Whitespace normalization
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Fullwidth to halfwidth
    full_to_half = {}
    for i in range(0xFF10, 0xFF19 + 1):
        full_to_half[i] = chr(i - 0xFEE0)
    for i in range(0xFF21, 0xFF3A + 1):
        full_to_half[i] = chr(i - 0xFEE0)
    for i in range(0xFF41, 0xFF5A + 1):
        full_to_half[i] = chr(i - 0xFEE0)

    result = []
    for ch in text:
        result.append(full_to_half.get(ord(ch), ch))
    text = "".join(result)

    # Step 5: Noise paragraph marking
    for i, para in enumerate(text.split("\n")):
        para_stripped = para.strip()
        if (re.match(r"^[\W_]+$", para_stripped) or
                re.match(r"^\d+$", para_stripped)):
            meta["noise_paragraphs"].append(i)

    return text, meta


# ============================================================
# Layer 2: Coarse Sentence Splitting
# ============================================================

def coarse_split(text: str, quote_ranges: list) -> list:
    """
    Returns: [{'text': ..., 'start': ..., 'end': ..., 'paragraph_id': ...}, ...]
    """
    paragraphs = text.split("\n")
    sentences = []

    char_offset = 0
    for pid, para in enumerate(paragraphs):
        if not para.strip():
            char_offset += len(para) + 1
            continue

        # Split on sentence terminators, protecting quotes and numbers
        raw_splits = _split_sentences(para, quote_ranges)

        for s_text, s_start_in_para in raw_splits:
            s_text = s_text.strip()
            if len(s_text) < 2:
                continue

            # Truncation: cap at 150 chars, split on commas
            if len(s_text) > 150:
                sub_parts = _truncate_long(s_text)
                for sp in sub_parts:
                    if len(sp.strip()) >= 2:
                        sentences.append({
                            "text": sp.strip(),
                            "start": char_offset + s_start_in_para,
                            "end": char_offset + s_start_in_para + len(sp),
                            "paragraph_id": pid,
                            "is_truncated": True,
                        })
            else:
                sentences.append({
                    "text": s_text,
                    "start": char_offset + s_start_in_para,
                    "end": char_offset + s_start_in_para + len(s_text),
                    "paragraph_id": pid,
                    "is_truncated": False,
                })

        char_offset += len(para) + 1

    return sentences


def _split_sentences(text: str, quote_ranges: list) -> list:
    """Split on sentence terminators, skip those inside quotes."""
    terminators = set("\u3002\uff01\uff1f\uff1b")  # . ! ? ;
    in_quote = {}
    for s, e in quote_ranges:
        for i in range(s, e):
            in_quote[i] = True

    splits = []
    last = 0
    for i, ch in enumerate(text):
        if ch in terminators and not in_quote.get(i):
            # Protect ellipsis and decimal points
            if ch == "\u3002" and i >= 2 and text[i - 2:i] == "\u3002\u3002":
                continue
            if ch == "\u3002" and i > 0 and i < len(text) - 1:
                if text[i - 1].isdigit() and text[i + 1].isdigit():
                    continue
            splits.append((text[last:i + 1], last))
            last = i + 1

    if last < len(text):
        splits.append((text[last:], last))

    return splits if splits else [(text, 0)]


def _truncate_long(text: str, max_len: int = 150) -> list:
    """Split long sentence on commas."""
    if len(text) <= max_len:
        return [text]
    parts = []
    for segment in re.split(r"(\uff0c)", text):  # fullwidth comma
        parts.append(segment)
    return parts if parts else [text]


# ============================================================
# Layer 3: Fine-Grained Splitting (Rule-based + spaCy)
# ============================================================

# Causal conjunctions (REVISED: split at these, tag causal_links)
CAUSAL_CONJ = re.compile(
    r"因为|所以|由于|因此|因而|之所以|是因为|"
    r"导致|造成|引起|促使|从而|进而|以至于"
)

# Transition markers
TRANSITION = re.compile(r"(但是|然而|可是|不过|却|虽然|尽管|但)")

# Attributive "的" protection: don't split "A的B"
ATTRIBUTIVE_DE = re.compile(r"(\S{1,8}的\S{1,8})")


def fine_split(sentences: list, nlp=None) -> list:
    """
    Step 1: Rule-based split (coordination, transition, causal REVISED)
    Step 2: Dependency parse each fragment
    Returns: list of claim dicts with causal_links populated
    """
    claims = []
    claim_id_counter = [0]  # mutable counter

    for sent in sentences:
        text = sent["text"]
        base_id = claim_id_counter[0]

        # Phase A: Causal splitting (REVISED)
        # Split at causal conjunctions -> cause-side and effect-side
        # Each side internally goes through coordination + transition
        segments, causal_markers = _causal_split(text)

        # Phase B: For each segment, apply coordination + transition
        all_fragments = []
        seg_boundaries = []  # track which fragments belong to which segment
        for seg in segments:
            start_idx = len(all_fragments)
            subs = _split_coordination_v2(seg)
            for sub in subs:
                trans_subs = _split_transition(sub)
                for ts in trans_subs:
                    if len(ts.strip()) >= 4:
                        all_fragments.append(ts.strip())
            seg_boundaries.append((start_idx, len(all_fragments)))

        if not all_fragments:
            continue

        # Phase C: Build claim dicts with spaCy
        for fi, frag in enumerate(all_fragments):
            claim = _build_claim(frag, sent, claim_id_counter[0], nlp)
            claims.append(claim)
            claim_id_counter[0] += 1

        # Phase D: Tag causal links (always cause_seg -> effect_seg)
        for cause_seg, effect_seg, marker in causal_markers:
            if cause_seg < len(seg_boundaries) and effect_seg < len(seg_boundaries):
                c_start, c_end = seg_boundaries[cause_seg]
                e_start, e_end = seg_boundaries[effect_seg]
                for ci in range(c_start, c_end):
                    for ei in range(e_start, e_end):
                        cid = base_id + ci
                        eid = base_id + ei
                        if cid < len(claims) and eid < len(claims):
                            claims[cid].setdefault("causal_links", [])
                            claims[eid].setdefault("causal_links", [])
                            claims[cid]["causal_links"].append({
                                "direction": "cause",
                                "target_claim_id": eid,
                                "marker": marker,
                            })
                            claims[eid]["causal_links"].append({
                                "direction": "effect",
                                "target_claim_id": cid,
                                "marker": marker,
                            })

    return claims


# Causal marker categories (sans capture groups for CAUSAL_CONJ)
CAUSE_MARKERS = {"因为", "由于", "既然", "鉴于"}
EFFECT_MARKERS = {"所以", "因此", "因而", "导致", "造成", "引起", "促使", "从而", "进而", "以至于"}
ALL_CAUSAL = CAUSE_MARKERS | EFFECT_MARKERS

def _causal_split(text: str) -> tuple:
    """
    Split at causal conjunctions using regex scanning.
    Returns (segments, markers) where markers are (cause_idx, effect_idx, conjunction).
    """
    # Find all causal marker positions
    positions = []
    for m in CAUSAL_CONJ.finditer(text):
        positions.append((m.start(), m.end(), m.group()))

    if not positions:
        return [text], []

    segments = []
    markers = []
    last_end = 0

    for start, end, marker_text in positions:
        # Text before this marker is a segment
        before = text[last_end:start].strip()
        if before:
            segments.append(before)

        last_end = end

        if marker_text in CAUSE_MARKERS:
            # '因为X' -> X (the segment before) is cause of the next segment
            # But X isn't added yet if this is the first marker and nothing before it
            cause_idx = len(segments) - 1 if segments else -1
            # effect will be added next
            markers.append((cause_idx, len(segments), marker_text))
        elif marker_text in EFFECT_MARKERS:
            # '所以Y' or '导致Y' -> previous segment caused Y
            cause_idx = len(segments) - 1 if segments else -1
            markers.append((cause_idx, len(segments), marker_text))

    # Remaining text after last marker
    after = text[last_end:].strip()
    if after:
        segments.append(after)

    # Fix: for EFFECT_MARKERS, the effect segment index might need adjustment
    # Re-process: for each marker, determine cause/effect segment indices
    # Strategy: scan segments and markers to build clean pairs
    resolved = []
    for cause_idx, _, marker_text in markers:
        if cause_idx >= 0:
            # effect is cause_idx + 1 (the segment after the marker)
            effect_idx = cause_idx + 1
            if effect_idx < len(segments):
                resolved.append((cause_idx, effect_idx, marker_text))

    return segments, resolved


def _split_coordination_v2(text: str) -> list:
    """
    Split A, B, and C into [A, B, C].
    Protects attributive '的' constructions: 'A的B' stays together.
    """
    # Step 1: Protect '的' constructions by replacing delimiter
    protected = ATTRIBUTIVE_DE.sub(lambda m: m.group(0).replace('、', '\x00'), text)

    # Step 2: Split on 、and ，
    if '、' not in protected and '，' not in protected:
        return [text]

    parts = re.split(r"[、，]", protected)
    result = []
    for p in parts:
        p = p.replace('\x00', '、').strip()
        # Remove trailing conjunctions
        p = re.sub(r"[和及与]$", "", p)
        # Remove leading/trailing punctuation debris
        p = re.sub(r"^[，,。.]+", "", p)
        p = re.sub(r"[，,。.]+$", "", p)
        if len(p) >= 4:
            result.append(p)

    return result if result else [text]


def _split_transition(text: str) -> list:
    """Split on transition markers: A，但是B -> [A, B]."""
    if not TRANSITION.search(text):
        return [text]

    parts = TRANSITION.split(text)
    result = []
    for p in parts:
        p = p.strip()
        if not p or TRANSITION.fullmatch(p):
            continue
        # Clean up punctuation debris
        p = re.sub(r"^[，,。.]+", "", p)
        p = re.sub(r"[，,。.]+$", "", p)
        if len(p) >= 4:
            result.append(p)
    return result if result else [text]


def _build_claim(frag: str, sent: dict, claim_id: int, nlp=None) -> dict:
    """Build a claim dict from a text fragment with spaCy metadata."""
    claim = {
        "claim_text": frag,
        "claim_id": claim_id,
        "char_offset_start": sent["start"],
        "char_offset_end": sent["end"],
        "paragraph_id": sent["paragraph_id"],
        "entity_snapshot": [],
        "time_expressions": [],
        "source_info": None,
        "core_verb": None,
        "is_speculation": False,
        "is_truncated": sent.get("is_truncated", False),
        "is_incomplete": False,
        "transition_pair_id": None,
        "causal_links": [],
        "has_number": False,
    }

    # Number detection
    if re.search(r"\d+", frag):
        claim["has_number"] = True

    # Source attribution
    src_match = re.search(r"据([^，,。]+?)(?:报道|披露|显示|称|表示)", frag)
    if src_match:
        claim["source_info"] = src_match.group(0)

    # Speculation
    if re.search(r"(可能|预计|或将|有望|大概率|不排除|或会|料将)", frag):
        claim["is_speculation"] = True

    # POS analysis: spaCy preferred, jieba fallback
    if nlp:
        try:
            doc = nlp(frag)
            roots = [t for t in doc if t.dep_ == "ROOT"]
            if roots:
                claim["core_verb"] = roots[0].text
            else:
                claim["is_incomplete"] = True
            for ent in doc.ents:
                claim["entity_snapshot"].append(ent.text)
                if ent.label_ in ("DATE", "TIME"):
                    claim["time_expressions"].append(ent.text)
        except Exception:
            claim["is_incomplete"] = True
    else:
        # jieba POS fallback
        try:
            import jieba.posseg as pseg
            words = list(pseg.cut(frag))
            verbs = [(w, f) for w, f in words if f.startswith('v')]
            if verbs:
                claim["core_verb"] = verbs[0][0]
            else:
                claim["is_incomplete"] = True
            # Simple entity extraction: nr/ns/nz/n t tags
            for w, f in words:
                if f in ('nr', 'ns', 'nz', 'nt'):
                    claim["entity_snapshot"].append(w)
                if f == 't' or (f == 'm' and ('年' in w or '月' in w or '日' in w)):
                    claim["time_expressions"].append(w)
            # Also catch explicit date patterns
            import re as _re
            for m in _re.finditer(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{4}年', frag):
                if m.group() not in claim["time_expressions"]:
                    claim["time_expressions"].append(m.group())
        except Exception:
            claim["is_incomplete"] = True

    return claim


# Layer 4: Metadata Binding + SimHash Dedup
# ============================================================

SIMHASH_BITS = 64


def _simhash(text: str) -> int:
    """Compute SimHash fingerprint for a text."""
    # Simple token-based SimHash
    tokens = list(text)
    if not tokens:
        return 0

    v = [0] * SIMHASH_BITS
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(SIMHASH_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fp = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            fp |= (1 << i)
    return fp


def _hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two SimHash fingerprints."""
    x = a ^ b
    return x.bit_count()


def bind_metadata(claims: list, original_text: str, nlp=None) -> list:
    """
    Bind metadata: pronoun resolution, entity completion.
    Then: SimHash dedup.
    Returns deduplicated claims list.
    """
    if not claims:
        return []

    # Pronoun resolution: within same paragraph, "该公司" -> previous claim's main entity
    para_entities = {}
    for claim in claims:
        pid = claim["paragraph_id"]
        if pid not in para_entities:
            para_entities[pid] = []

        text = claim["claim_text"]
        # Resolve pronouns
        for pronoun in ["该公司", "该企业", "其", "该机构"]:
            if pronoun in text and para_entities[pid]:
                text = text.replace(pronoun, para_entities[pid][-1])
                claim["claim_text"] = text

        # Track entity for next claim in same paragraph
        if claim["entity_snapshot"]:
            para_entities[pid].append(claim["entity_snapshot"][0])

    # --- SimHash Dedup ---
    seen = {}  # fingerprint -> claim
    deduped = []
    for claim in claims:
        fp = _simhash(claim["claim_text"])
        is_dup = False
        for existing_fp, existing_claim in list(seen.items()):
            if _hamming_distance(fp, existing_fp) <= 3:
                # Merge: append source if different
                if "merged_sources" not in existing_claim:
                    existing_claim["merged_sources"] = [existing_claim["claim_text"]]
                existing_claim["merged_sources"].append(claim["claim_text"])
                is_dup = True
                break
        if not is_dup:
            seen[fp] = claim
            deduped.append(claim)

    return deduped


# ============================================================
# Main Atomizer
# ============================================================


# Hollow reporting patterns that shouldn't be standalone claims
HOLLOW_REPORTERS = re.compile(
    r"^(会议|报告|声明|公告|通知|"
    r"强调|指出|表示|认为|"
    r"发布|披露|确认|声称|"
    r"宣称|提出|说明|介绍)"
    r"[\uff0c\u3002]*$"
)

def _merge_hollow_clauses(claims):
    """
    Merge short hollow reporting clauses with the following claim.
    e.g. ["会议指出", "要纵深推进"] -> ["会议指出，要纵深推进"]
    Only merges when the first claim is a short (<10 char) hollow reporter.
    """
    if len(claims) < 2:
        return claims
    
    merged = []
    skip_next = False
    
    for i, c in enumerate(claims):
        if skip_next:
            skip_next = False
            continue
        
        text = c["claim_text"]
        next_c = claims[i + 1] if i + 1 < len(claims) else None
        
        # Check if this is a hollow reporter that should merge with next
        if (next_c and len(text) <= 10 and 
            HOLLOW_REPORTERS.match(text) and 
            len(next_c["claim_text"]) > 3):
            # Merge: use comma separator
            merged_text = text + "，" + next_c["claim_text"]
            new_claim = dict(c)  # copy
            new_claim["claim_text"] = merged_text
            new_claim["char_offset_end"] = next_c.get("char_offset_end", c.get("char_offset_end", 0))
            new_claim["is_incomplete"] = False
            merged.append(new_claim)
            skip_next = True
        else:
            merged.append(c)
    
    return merged


def atomize(text: str, is_html: bool = False, nlp=None) -> list:
    """
    Full L0 atomization pipeline.
    Returns: list of atomic claim dicts.
    """
    # L1: Preprocess
    clean_text, meta = preprocess(text, is_html=is_html)

    # L2: Coarse split
    sentences = coarse_split(clean_text, meta["quote_ranges"])

    # L3: Fine split (use spaCy if available)
    actual_nlp = nlp if nlp is not None else _NLP
    claims = fine_split(sentences, nlp=actual_nlp)

    # L4: Metadata binding + dedup
    claims = bind_metadata(claims, clean_text, nlp=actual_nlp)

    # Post-processing: merge hollow reporting clauses with their content
    # e.g. "会议指出" + "要纵深推进相关制度建设" -> "会议指出，要纵深推进相关制度建设"
    claims = _merge_hollow_clauses(claims)

    return claims
