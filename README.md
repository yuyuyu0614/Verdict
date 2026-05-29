# Verdict

国内首个零 LLM、确定性输出的中文信息去噪引擎。噪声率 0.1%，纯 CPU <300MB，无需 GPU，无需 API Key。

```
L0 atomize -> L0.5 noise filter (5-layer rules + LR fusion)
```

## Performance

| Metric | Value |
|--------|-------|
| Noise rate | 0.104% (verified on 17,232 Chinese news articles) |
| False positive rate | 0% |
| LR accuracy | 80.3% (hybrid: jieba + spaCy) |
| Processing speed | ~36 claims/s |
| Memory | <300MB (including spaCy 45MB) |
| LLM calls | 0 |
| Tested data | 17,000+ real Chinese news |

## vs. Alternatives

| | Verdict | Tencent JiaoZhen | GPT-4 Fact Check | ClaimBuster |
|---|---|---|---|---|
| Tech | Rules + ML | GPT + Editors | LLM API | BERT + GPU |
| Chinese | ✅ Native | ✅ | ❌ | ❌ |
| Cost per query | ¥0 | Per token | Per token | GPU cost |
| Deterministic | ✅ 100% | ❌ Black-box | ❌ | ❌ |
| Open source | ✅ MIT | ❌ | ❌ | ✅ |
| On-premise | ✅ <300MB | ❌ | ❌ | >2GB |

## Quick Start

```bash
pip install -r requirements.txt
python -c "from src.l0_engine.noise_engine.pipeline import filter_noise; print('OK')"
```

## API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8002
curl -X POST http://127.0.0.1:8002/evaluate_v2 -H 'Content-Type: application/json' -d '{"text": "恒大集团负债2.4万亿元"}'
```

## Use Cases

- **Financial news**: Filter stale data (stock prices, market caps) and spam
- **Academic writing**: Verify cited data is traceable and up-to-date
- **Policy analysis**: Detect hollow diplomatic rhetoric
- **Agent middleware**: Use as input filter for any Chinese LLM agent

## Architecture

```
claim_text
  |
  v
L1: Ad/spam keyword filter  --> discard
  |
  v
Reporter glue pattern match --> discard
  |
  v
L2: Structural hollowness   --> discard (>= 0.7)
  |
  v
L3: Causal risk analysis
  |
  v
L4: Temporal staleness
  |
  v
L5: Source scoring
  |
  v
LR Fusion: predict_noise_score -> classify_noise
  |
  v
pass / review / discard
```

## License

MIT
