# L4: Logic Validation
# Temporal staleness check + intra-document contradiction detection

import re
from datetime import datetime

# Volatile attributes: how long until data is considered stale
VOLATILE_ATTRS = {
    "注册资本": {"max_days": 730, "category": "企业信息"},
    "总资产": {"max_days": 365, "category": "财务"},
    "负债": {"max_days": 365, "category": "财务"},
    "营收": {"max_days": 365, "category": "财务"},
    "净利润": {"max_days": 365, "category": "财务"},
    "员工人数": {"max_days": 365, "category": "企业信息"},
    "市值": {"max_days": 30, "category": "金融"},
    "股价": {"max_days": 7, "category": "金融"},
    "市盈率": {"max_days": 30, "category": "金融"},
    "法定代表人": {"max_days": 1095, "category": "企业信息"},
    "注册地址": {"max_days": 1095, "category": "企业信息"},
    "股东": {"max_days": 365, "category": "企业信息"},
    "高管": {"max_days": 365, "category": "企业信息"},
    "存续状态": {"max_days": 365, "category": "企业信息"},
    "注册资本变更": {"max_days": 730, "category": "企业信息"},
    "实缴资本": {"max_days": 730, "category": "企业信息"},
    "经营范围": {"max_days": 1095, "category": "企业信息"},
    "统一社会信用代码": {"max_days": 9999, "category": "固定信息"},
    "成立日期": {"max_days": 9999, "category": "固定信息"},
    "债券评级": {"max_days": 90, "category": "金融"},
    "GDP": {"max_days": 90, "category": "宏观"},
    "CPI": {"max_days": 30, "category": "宏观"},
    "失业率": {"max_days": 30, "category": "宏观"},
    "基准利率": {"max_days": 30, "category": "宏观"},
    "外汇储备": {"max_days": 30, "category": "宏观"},
    "房价指数": {"max_days": 30, "category": "宏观"},
    "人口": {"max_days": 365, "category": "宏观"},
    "土地面积": {"max_days": 3650, "category": "地理"},
}


def temporal_staleness(claim_text: str, doc_date: str = None) -> float:
    """
    Check if claim contains stale attribute values.
    Compares claim year against reference date or current date.
    Returns [0-1], higher = more stale.
    """
    if not doc_date:
        ref_year = datetime.now().year
    else:
        try:
            ref_year = int(doc_date[:4])
        except (ValueError, TypeError):
            ref_year = datetime.now().year

    for attr, config in VOLATILE_ATTRS.items():
        if attr in claim_text:
            times = re.findall(r"(\d{4})年", claim_text)
            if not times:
                return 0.0

            claim_year = int(times[0])
            gap_years = ref_year - claim_year

            if gap_years * 365 > config["max_days"]:
                return min(0.3 + 0.1 * gap_years, 1.0)
            break

    return 0.0


def contradiction_check(claims: list) -> dict:
    """
    Intra-document contradiction detection (experimental).
    State machine: proposed→approved/rejected→in_progress/cancelled→completed/dead.
    V1: empty implementation, returns empty dict.
    Future: entity state machine + conflict detection.
    """
    return {"conflicts": [], "incomplete": []}
