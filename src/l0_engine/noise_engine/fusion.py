# Fusion Layer: Logistic Regression Noise Scoring

import os
import pickle

# Default thresholds
NOISE_THRESHOLD = 0.70    # >= this -> discard
REVIEW_THRESHOLD = 0.30   # > this -> flag for review

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "models", "lr_noise.pkl")

_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            with open(_MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        except FileNotFoundError:
            # No trained model yet: return None, caller uses fallback
            pass
    return _model


def predict_noise_score(feature_vector: list) -> float:
    """
    Predict noise probability [0-1] using logistic regression.
    Falls back to simple weighted average if no model trained.
    """
    model = None  # DISABLED: overfitted, retrain needed
    if model is not None:
        try:
            proba = model.predict_proba([feature_vector])[0]
            model_score = float(proba[1])
            # Blend: model primary, hollowness as floor
            d7 = feature_vector[6]
            blended = max(model_score, d7 * 0.7)  # hollowness sets floor
            return round(min(blended, 1.0), 4)
        except Exception:
            pass

    # Fallback: conservative weighted average
    # Key signals: hollowness(D7) = +noise, context_quality(D10) = -noise
    # causal(D8) indicates factual content, temporal(D9) indicates staleness
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12 = feature_vector
    
    # Hollowness-driven: if hollowness > 0, use it directly as noise floor
    if d7 > 0:
        # Blend hollowness with context: high hollowness + low context = noise
        fallback = 0.6 * d7 + 0.4 * (1.0 - d10)
    else:
        fallback = 0.0
    
    # Clamp
    fallback = min(fallback, 1.0)
    return round(fallback, 4)


def classify_noise(noise_score: float) -> str:
    """
    Returns: 'discard', 'review', or 'pass'
    """
    if noise_score >= NOISE_THRESHOLD:
        return "discard"
    elif noise_score > REVIEW_THRESHOLD:
        return "review"
    else:
        return "pass"
