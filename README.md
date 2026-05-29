# Verdict

Zero-LLM noise filtering pipeline for Chinese news claim extraction.

```
L0 atomize -> L0.5 noise filter (5-layer rules + LR fusion)
```

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
