# ai_engine.py
# Explainable heuristic "AI" engine, entry/target/stop computation, persistence for correct predictions.

import math
import json
from pathlib import Path
from typing import Dict, Any

RECORDS_FILE = Path("prediction_records.json")

def logistic(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def normalize(v, lo, hi):
    if v is None:
        return 0.0
    try:
        v = float(v)
    except Exception:
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def features_from_indicator_snapshot(indicators: Dict[str, Any]) -> Dict[str, float]:
    f = {}
    score = indicators.get('score', 0)
    f['score_norm'] = normalize(score, -100, 100)
    rsi = indicators.get('rsi14', None)
    f['rsi_norm'] = normalize(rsi, 10, 90)
    macd = abs(indicators.get('macd_hist', 0))
    f['macd_norm'] = normalize(macd, 0, max(1.0, macd*1.5))
    vol_osc = indicators.get('vol_osc', 0)
    f['vol_norm'] = normalize(vol_osc, -1, 1)
    atr = indicators.get('atr14', 0)
    price = indicators.get('price', 1.0)
    f['volatility'] = normalize((atr / (price + 1e-9)), 0.0, 0.1)
    nw = indicators.get('nw_slope', 0)
    # slope normalization relative to price
    f['nw_norm'] = normalize(nw, -price*0.01, price*0.01)
    return f

def predict_probability(indicators: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
    if weights is None:
        weights = {
            'score': 2.5,
            'rsi': 1.8,
            'macd': 1.5,
            'vol': 1.2,
            'nw': 1.4,
            'volatility_penalty': -2.0
        }
    feats = features_from_indicator_snapshot(indicators)
    x = 0.0
    x += weights['score'] * (feats['score_norm'] - 0.5)
    x += weights['rsi'] * (0.5 - feats['rsi_norm'])
    x += weights['macd'] * (feats['macd_norm'] - 0.2)
    x += weights['vol'] * (feats['vol_norm'] - 0.2)
    x += weights['nw'] * (feats['nw_norm'] - 0.2)
    x += weights['volatility_penalty'] * (feats['volatility'] - 0.2)
    prob = logistic(x)
    explanation = {
        'features': feats,
        'raw_score': x,
        'probability': prob,
        'text': (
            f"Prob {prob:.2f} — score_norm={feats['score_norm']:.2f}, rsi_norm={feats['rsi_norm']:.2f}, "
            f"macd_norm={feats['macd_norm']:.2f}, vol_norm={feats['vol_norm']:.2f}, nw_norm={feats['nw_norm']:.2f}, vol={feats['volatility']:.2f}"
        )
    }
    return explanation

def compute_entry_target_stop(price: float, risk_pct: float = 0.5, target_reward_ratio: float = 2.0, atr: float = None):
    entry = float(price)
    if atr is not None and atr > 0:
        stop_distance = 1.5 * float(atr)
        stop = entry - stop_distance
    else:
        stop = entry * (1.0 - risk_pct / 100.0)
        stop_distance = entry - stop
    if stop_distance <= 0:
        stop_distance = entry * 0.005
        stop = entry - stop_distance
    target = entry + stop_distance * target_reward_ratio
    return {'entry': entry, 'stop': max(0.0, stop), 'target': target, 'stop_distance': stop_distance}

def load_records():
    if not RECORDS_FILE.exists():
        return []
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_record(record: Dict[str, Any]):
    recs = load_records()
    recs.append(record)
    try:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def clear_records():
    try:
        if RECORDS_FILE.exists():
            RECORDS_FILE.unlink()
        return True
    except Exception:
        return False
