# L5: Source Quality Scoring

def source_score(source_url=None, source_name=None):
    if not source_name and not source_url:
        return 0.5
    if source_name:
        high = ["新华社", "人民日报", "央视", "财新", "证监会", "央行", "国家统计局"]
        for h in high:
            if h in source_name:
                return 0.95
        med = ["新浪", "腾讯", "网易", "搜狐", "澎湃", "界面"]
        for m in med:
            if m in source_name:
                return 0.7
    if source_url and ".gov.cn" in source_url:
        return 0.95
    return 0.5

