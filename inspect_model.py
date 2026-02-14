"""
Inspect what words the model learned are most toxic.
This helps understand feature importance in your ML model!
"""

import joblib
import numpy as np

# Load your trained model
print("📂 Loading model...\n")
model_data = joblib.load('model/model.joblib')
vectorizer = model_data['vectorizer']
model = model_data['model']

# Get feature names (words and word pairs)
feature_names = vectorizer.get_feature_names_out()
print(f"Model knows {len(feature_names)} words/phrases\n")

# Get coefficients (importance scores)
# Positive = toxic, Negative = non-toxic
coefficients = model.coef_[0]

print("="*60)
print("🔥 TOP 20 MOST TOXIC WORDS/PHRASES:")
print("="*60)

# Top toxic words
top_toxic_idx = np.argsort(coefficients)[-20:][::-1]
for i, idx in enumerate(top_toxic_idx, 1):
    word = feature_names[idx]
    score = coefficients[idx]
    print(f"{i:2d}. {word:25s} → {score:.3f}")

print("\n" + "="*60)
print("✅ TOP 20 LEAST TOXIC (MOST POSITIVE) WORDS:")
print("="*60)

# Top non-toxic words
top_nontoxic_idx = np.argsort(coefficients)[:20]
for i, idx in enumerate(top_nontoxic_idx, 1):
    word = feature_names[idx]
    score = coefficients[idx]
    print(f"{i:2d}. {word:25s} → {score:.3f}")

print("\n" + "="*60)
print("🎓 LEARNING NOTES:")
print("="*60)
print("• Positive scores → model associates with TOXIC")
print("• Negative scores → model associates with NON-TOXIC")
print("• Larger magnitude = stronger association")
print("• These are the 'features' your Logistic Regression learned!")
print("="*60)
