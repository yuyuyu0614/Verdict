# L2: Information Theory Pre-filter (paragraph-level)
# Computes context quality before L0 atomization

import math, gzip
from collections import Counter

try:
    import jieba
except ImportError:
    jieba = None


def compute_context_quality(text: str) -> float:
    """
    Paragraph-level information quality score [0-1].
    Higher = more informative. Lower = likely boilerplate/vacuum.
    Combines: lexical entropy (50%) + gzip compression ratio (50%).
    """
    if len(text) < 30:
        return 0.5  # single sentence: neutral default

    # Lexical entropy via jieba segmentation
    if jieba:
        words = list(jieba.cut(text))
    else:
        words = list(text)

    freq = Counter(words)
    total = len(words)

    if total <= 1:
        return 0.5

    entropy = -sum((f / total) * math.log2(f / total) for f in freq.values())
    max_entropy = math.log2(total)
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0.5

    # Gzip compression ratio
    compressed = gzip.compress(text.encode("utf-8"))
    ratio = len(compressed) / len(text.encode("utf-8"))
    # ratio ~1.0 = random/incompressible (high info)
    # ratio < 0.8 = highly repetitive (low info)
    compression_score = max(0.0, min(1.0, (0.85 - ratio) / 0.25))

    return round(0.5 * entropy_score + 0.5 * compression_score, 4)
