"""
Training script for toxicity classification model.
This trains a simple but effective model using TF-IDF + Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
import json
import os

def load_data(data_path):
    """Load the Jigsaw toxicity dataset"""
    print(f"📂 Loading data from {data_path}...")
    # Handle malformed rows by skipping bad lines
    df = pd.read_csv(data_path, encoding='latin-1', on_bad_lines='skip', engine='python')
    
    # Keep only text and toxic column
    df = df[['comment_text', 'toxic']].copy()
    df.columns = ['text', 'is_toxic']
    
    # Convert toxic column to integer (handle any string values)
    df['is_toxic'] = pd.to_numeric(df['is_toxic'], errors='coerce').fillna(0).astype(int)
    
    print(f"✅ Loaded {len(df)} comments")
    print(f"   Toxic: {df['is_toxic'].sum()} ({df['is_toxic'].mean()*100:.1f}%)")
    print(f"   Non-toxic: {(df['is_toxic'] == 0).sum()}")
    
    return df

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def train_model(df, max_samples=50000):
    """Train TF-IDF + Logistic Regression model"""
    
    # Limit size for faster training
    if len(df) > max_samples:
        print(f"⚡ Sampling {max_samples} comments for faster training...")
        df = df.sample(max_samples, random_state=42)
    
    # Clean text
    print("🧹 Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Split data: 80% train, 20% test
    print("✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['is_toxic'], 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # TF-IDF Vectorization
    print("\n🔤 Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        strip_accents='unicode',
        lowercase=True,
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Vectorizer created with {len(vectorizer.vocabulary_)} features")
    
    # Train Logistic Regression
    print("\n🤖 Training Logistic Regression model...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_vec, y_train)
    print("✅ Model trained!")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-toxic', 'Toxic']))
    
    return vectorizer, model, {'f1': f1, 'auc': auc}

def save_model(vectorizer, model, metrics, output_dir='model'):
    """Save the trained model and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump({'vectorizer': vectorizer, 'model': model}, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    meta = {
        'metrics': metrics,
        'model_type': 'TfidfVectorizer + LogisticRegression',
        'features': len(vectorizer.vocabulary_)
    }
    
    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"📄 Metadata saved to {meta_path}")

def main():
    """Main training pipeline"""
    print("🚀 Starting Training Pipeline\n")
    
    data_path = 'data/train.csv'
    df = load_data(data_path)
    
    vectorizer, model, metrics = train_model(df, max_samples=50000)
    
    save_model(vectorizer, model, metrics)
    
    print("\n✅ Training complete! Model ready to use.")
    print("\nNext steps:")
    print("1. Run: streamlit run app/final_streamlit_app.py")

if __name__ == '__main__':
    main()
