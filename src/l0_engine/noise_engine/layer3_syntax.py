# L3: Dependency Syntax Analysis (jieba POS fallback)
# Uses jieba part-of-speech tagging when spaCy is unavailable

import re

# Known entities that jieba might miss (industries, concepts, event names)
KNOWN_ENTITIES = {
    # Industries/sectors
    "低空经济", "商业航天", "量子计算", "人工智能", "新能源汽车",
    "半导体", "光伏", "储能", "氢能", "生物医药", "数字经济",
    "跨境电商", "智能制造", "机器人", "元宇宙", "碳中和",
    "油田", "天然气", "煤炭", "风电", "核电", "水电",
    "物流", "快递", "外卖", "网约车", "共享单车",
    # Financial concepts
    "IPO", "ETF", "REITs", "公募基金", "私募基金", "科创板",
    "创业板", "北交所", "主板", "新三板",
    # Policy/law terms
    "反垄断", "合规管理", "注册制", "碳排放权", "数据要素",
    "劳动者", "权益保障", "新规", "立法", "管理条例",
    # Country/region (common in news)
    "中塞", "中巴", "中美", "中日", "中欧", "中俄", "中法",
    "一带一路", "RCEP", "CPTPP", "东盟", "金砖",
}

try:
    import jieba.posseg as pseg
    HAS_JIEBA_POS = True
except ImportError:
    HAS_JIEBA_POS = False

HOLLOW_VERBS = {
    # Truly empty diplomatic/reporting verbs
    "认为", "表示", "指出", "重申", "强调", "高度评价",
    "一致认为", "赞同", "呼吁",
    # Political boilerplate (only when dominating the sentence)
    "贯彻", "坚定", "增强", "践行",
    # Cooperation boilerplate (diplomatic, without concrete object)
    "携手",
}

# Broader verbs that only count as hollow when NO named entity present
CONDITIONAL_HOLLOW = {
    "推进", "加快", "深化", "全面", "提高", "强化", "坚持",
    "推动", "发展", "促进", "保障", "完善", "健全",
    "落实", "加强", "统筹", "协调", "优化", "提升",
    "主张", "宣称", "声称",
    "共建", "共享", "共商", "构建",
}

CAUSAL_MARKERS = {"因为", "所以", "由于", "导致", "造成", "引起", "促使", "取决于",
                  "受", "得益于", "归因于", "缘于", "因此", "因而"}

RESULT_VERBS = {"上涨", "下跌", "增长", "下降", "突破", "跌至", "飙升", "暴跌",
                "反弹", "回暖", "走高", "走低", "复苏", "回落"}


def structural_hollowness(claim_text: str, nlp=None) -> float:
    """
    Detects claims that look like facts but have no verifiable content.
    spaCy path: uses dependency depth + branching factor.
    jieba fallback: POS-based hollow verb ratio + structure complexity.
    Returns [0-1], higher = more hollow.
    """
    if nlp is not None:
        return _spacy_hollowness(claim_text, nlp)

    # jieba fallback
    if not HAS_JIEBA_POS:
        return 0.0

    words = list(pseg.cut(claim_text))
    if not words:
        return 0.0

    total = len(words)

    # Count hollow verbs (reporting verbs that indicate no real content)
    hollow_count = sum(1 for w, flag in words if w in HOLLOW_VERBS)
    # Conditional hollow: only count if no named entity present
    has_named_entity = (
        any(flag in ('nr', 'ns', 'nz', 'nt') for w, flag in words) or
        any(ent in claim_text for ent in KNOWN_ENTITIES)
    )
    cond_count = sum(1 for w, flag in words if w in CONDITIONAL_HOLLOW)
    if not has_named_entity:
        # No entity: all conditional hollow verbs count
        hollow_count += cond_count
    # If has entity: conditional hollow verbs are ignored (news context overrides)

    # Count all verbs and verbal nouns
    verb_count = sum(1 for w, flag in words if flag.startswith('v'))

    # Count content-bearing words
    content_count = sum(1 for w, flag in words
                        if flag.startswith(('n', 'v', 'a', 'm', 'q', 'ns', 'nr', 'nz')))

    # Count numbers (strong signal of factual content)
    has_number = any(flag.startswith('m') or w.isdigit() for w, flag in words)
    has_numeric = bool(re.search(r'\d+', claim_text))

    # Metrics
    hollow_ratio = hollow_count / max(verb_count, 1)
    content_ratio = content_count / max(total, 1)

    score = 0.0

    if hollow_count >= 1 and hollow_ratio >= 0.3:
        score += 0.5  # hollow verb dominates

    if content_ratio < 0.3 and not has_numeric:
        score += 0.3

    if verb_count == 0 and total >= 4 and not has_numeric:
        score += 0.2

    if total < 4 and not has_numeric:
        score += 0.1

    # Domain exemptions: legal/financial short sentences are NOT hollow
    legal_markers = {'法院', '裁定', '判决', '驳回', '受理', '破产', '重整', '仲裁', '原告', '被告'}
    financial_markers = {'营收', '净利润', '同比增长', '环比', '突破', '涨幅', '跌幅', '指数'}
    if any(m in claim_text for m in legal_markers):
        score = max(0.0, score - 0.3)
    if any(m in claim_text for m in financial_markers):
        score = max(0.0, score - 0.2)

    return min(score, 1.0)

def _spacy_hollowness(claim_text, nlp):
    """spaCy path for structural hollowness (kept separate for clarity)."""
    doc = nlp(claim_text)
    depth = max([len(list(t.ancestors)) for t in doc] + [0])
    branch_factor = (len([t for t in doc if t.dep_ != "ROOT" and t.head.i != t.i])
                     / max(len(doc), 1))
    has_hollow = any(t.text in HOLLOW_VERBS for t in doc)
    score = 0.0
    if depth <= 2:
        score += 0.5  # hollow verb dominates
    if branch_factor < 0.3:
        score += 0.3
    if has_hollow:
        score += 0.3
    return min(score, 1.0)


def causal_risk(claim_text: str, nlp=None, neighbor_claims: list = None) -> float:
    """
    Detects potential causal misattribution.
    spaCy path: dependency-based.
    jieba fallback: keyword matching + temporal check.
    Returns [0-1], higher = more suspicious.
    """
    if nlp is not None:
        doc = nlp(claim_text)
        has_causal = any(t.text in CAUSAL_MARKERS for t in doc)
        if not has_causal:
            if neighbor_claims:
                if any(t.text in RESULT_VERBS for t in doc):
                    return 0.15
            return 0.0
        times = re.findall(r"\d{4}年|\d{1,2}月\d{1,2}日", claim_text)
        if len(times) >= 2:
            return 0.3
        else:
            return 0.5

    # jieba fallback
    # Explicit causal marker detection
    has_explicit = any(m in claim_text for m in CAUSAL_MARKERS)

    if not has_explicit:
        # Implicit: result verbs without causal context
        if neighbor_claims:
            has_result = any(v in claim_text for v in RESULT_VERBS)
            if has_result:
                return 0.15
        return 0.0

    # Has causal marker: check temporal precision
    times = re.findall(r"\d{4}年|\d{1,2}月\d{1,2}日", claim_text)
    if len(times) >= 2:
        return 0.3  # causal with temporal alignment
    else:
        return 0.5  # causal without temporal anchor -> suspicious
