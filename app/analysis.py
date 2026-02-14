"""
Analysis functions for thread escalation prediction.
"""

import numpy as np
import pandas as pd

def parse_thread(text):
    """
    Parse pasted thread text into individual messages.
    
    Assumes format like:
    User1: message
    User2: message
    
    Or just line-by-line messages.
    """
    lines = text.strip().split('\n')
    messages = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to extract username if format is "User: message"
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                user = parts[0].strip()
                message = parts[1].strip()
            else:
                user = "Unknown"
                message = line
        else:
            user = "Unknown"
            message = line
        
        messages.append({'user': user, 'text': message})
    
    return messages

def score_messages(messages, model_dict):
    """
    Score each message for toxicity.
    
    Args:
        messages: List of message dicts with 'text' field
        model_dict: Dict with 'vectorizer' and 'model'
        
    Returns:
        List of scores (0-1, higher = more toxic)
    """
    vectorizer = model_dict['vectorizer']
    model = model_dict['model']
    
    texts = [msg['text'] for msg in messages]
    
    # Vectorize
    X = vectorizer.transform(texts)
    
    # Get probability of toxic class
    scores = model.predict_proba(X)[:, 1]
    
    return scores

def compute_escalation_features(scores, k=3):
    """
    Compute escalation features from toxicity scores.
    
    Args:
        scores: Array of toxicity scores
        k: Window size for moving average
        
    Returns:
        Dictionary of features
    """
    scores = np.array(scores)
    n = len(scores)
    
    if n == 0:
        return {
            'mean_toxicity': 0,
            'max_toxicity': 0,
            'trend_slope': 0,
            'volatility': 0,
            'high_toxicity_share': 0
        }
    
    # Basic stats
    mean_tox = float(np.mean(scores))
    max_tox = float(np.max(scores))
    
    # Trend: fit linear regression
    if n > 1:
        x = np.arange(n)
        slope = np.polyfit(x, scores, 1)[0]
    else:
        slope = 0
    
    # Volatility: standard deviation
    volatility = float(np.std(scores))
    
    # High toxicity share: % above 0.5
    high_share = float(np.mean(scores > 0.5))
    
    return {
        'mean_toxicity': mean_tox,
        'max_toxicity': max_tox,
        'trend_slope': slope,
        'volatility': volatility,
        'high_toxicity_share': high_share
    }

def calculate_escalation_risk(features):
    """
    Calculate overall escalation risk score (0-100).
    
    Simple weighted combination of features.
    """
    risk = 0
    
    # Mean toxicity (40% weight)
    risk += features['mean_toxicity'] * 40
    
    # Positive trend (30% weight)
    if features['trend_slope'] > 0:
        risk += min(features['trend_slope'] * 100, 30)
    
    # High toxicity share (20% weight)
    risk += features['high_toxicity_share'] * 20
    
    # Volatility (10% weight)
    risk += min(features['volatility'] * 10, 10)
    
    return min(max(risk, 0), 100)

def detect_trigger_messages(scores, threshold=0.2):
    """
    Detect messages that trigger escalation.
    
    Looks for largest positive jumps in toxicity.
    
    Args:
        scores: Array of toxicity scores
        threshold: Minimum jump to consider
        
    Returns:
        List of (index, jump_size) tuples
    """
    if len(scores) < 2:
        return []
    
    scores = np.array(scores)
    jumps = np.diff(scores)
    
    triggers = []
    for i, jump in enumerate(jumps):
        if jump > threshold:
            triggers.append((i + 1, float(jump)))  # i+1 because diff shifts indices
    
    # Sort by jump size
    triggers.sort(key=lambda x: x[1], reverse=True)
    
    return triggers

def forecast_next_messages(scores, n_forecast=3):
    """
    Forecast toxicity of next n messages using linear extrapolation.
    
    Args:
        scores: Array of historical toxicity scores
        n_forecast: Number of future messages to predict
        
    Returns:
        Array of predicted scores
    """
    if len(scores) < 2:
        # Not enough data, return current average
        return np.array([np.mean(scores)] * n_forecast)
    
    scores = np.array(scores)
    n = len(scores)
    
    # Fit linear trend
    x = np.arange(n)
    coeffs = np.polyfit(x, scores, 1)
    
    # Predict future
    future_x = np.arange(n, n + n_forecast)
    forecast = np.polyval(coeffs, future_x)
    
    # Clip to [0, 1]
    forecast = np.clip(forecast, 0, 1)
    
    return forecast
