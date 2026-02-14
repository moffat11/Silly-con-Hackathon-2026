"""
ThreadCast: Predict when online arguments explode!
Complete Streamlit application with personality-based recommendations.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Import analysis functions
from analysis import (
    parse_thread,
    score_messages,
    compute_escalation_features,
    calculate_escalation_risk,
    detect_trigger_messages,
    forecast_next_messages
)

# Page config
st.set_page_config(
    page_title="🔥 ThreadCast",
    page_icon="🔥",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    """Load the trained toxicity model"""
    model_path = Path('model/model.joblib')
    if not model_path.exists():
        return None
    
    model_dict = joblib.load(model_path)
    
    # Load metadata
    meta_path = Path('model/meta.json')
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    
    return model_dict, meta

# Initialize session state for personality profile
if 'personality_profile' not in st.session_state:
    st.session_state.personality_profile = None

# Sample threads
SAMPLE_THREADS = {
    "Mild Disagreement": """Alice: I think pineapple on pizza is actually pretty good
Bob: I respectfully disagree, but to each their own
Alice: Fair enough, we all have different tastes""",
    
    "Heated Debate": """User1: This policy is completely wrong
User2: Actually you don't understand the economics
User1: No, YOU don't understand basic logic
User2: That's such a stupid take
User1: Are you seriously calling me stupid?
User2: If the shoe fits moron""",
    
    "Toxic Escalation": """Person1: I disagree with this article
Person2: You're completely wrong about that
Person1: No I'm not, read it again
Person2: You're an idiot if you believe this garbage
Person1: Don't call me an idiot you jerk
Person2: Shut up loser, nobody cares what you think
Person1: Go to hell you worthless piece of trash"""
}

def get_recommendation(risk_score, profile):
    """Generate personalized recommendation based on profile and risk"""
    
    if profile is None:
        return "💡 Complete your profile in the sidebar for personalized recommendations!"
    
    age = profile['age']
    purpose = profile['purpose']
    comfort = profile['comfort_level']
    
    recommendations = []
    
    # Age-based filtering
    if age == "Under 13" and risk_score > 50:
        recommendations.append("🚨 **WARNING**: This conversation contains language not suitable for your age group. We recommend avoiding this thread.")
        return recommendations
    
    elif age == "13-17" and risk_score > 70:
        recommendations.append("⚠️ **Caution**: This thread is getting very heated. Consider taking a break.")
    
    # Purpose-based advice
    if purpose == "Just browsing/scrolling":
        if risk_score > 60:
            recommendations.append("📱 **Scroll Past**: This thread is likely to waste your time and energy. Keep scrolling!")
        else:
            recommendations.append("✅ **Safe to Read**: This looks like a civil discussion.")
    
    elif purpose == "Engage in discussions":
        if risk_score > 70:
            recommendations.append("🛑 **Don't Engage**: This conversation has escalated. Your comment might fuel the fire.")
            recommendations.append("💭 **Alternative**: Try starting a fresh, respectful discussion on this topic instead.")
        elif risk_score > 40:
            recommendations.append("⚡ **Proceed with Caution**: This thread is heating up. If you engage:")
            recommendations.append("   • Stay respectful")
            recommendations.append("   • Focus on facts, not personal attacks")
            recommendations.append("   • Be prepared to disengage if it escalates")
        else:
            recommendations.append("💬 **Good Discussion**: This looks like a healthy debate. Feel free to contribute!")
    
    elif purpose == "Research/Learning":
        if risk_score > 50:
            recommendations.append("📚 **Research Note**: This thread shows escalation patterns. Good for studying online behavior, but consider the emotional toll.")
        else:
            recommendations.append("📖 **Learning Opportunity**: Clean discussion with good arguments on both sides.")
    
    # Comfort level adjustments
    if comfort == "Prefer clean language" and risk_score > 30:
        recommendations.append("🔇 **Language Warning**: This thread may contain profanity or insults that exceed your comfort level.")
    
    elif comfort == "Some casual language okay" and risk_score > 60:
        recommendations.append("💢 **Intensity Warning**: Language goes beyond casual into hostile territory.")
    
    # General advice for toxic threads
    if risk_score > 70:
        recommendations.append("\n**Why We Don't Recommend Engaging:**")
        recommendations.append("• Research shows toxic threads rarely change minds")
        recommendations.append("• They can negatively impact your mood for hours")
        recommendations.append("• Your mental health > winning an argument")
    
    if not recommendations:
        recommendations.append("✅ This looks like a respectful conversation. Enjoy!")
    
    return recommendations

# Sidebar - Personality Profile
with st.sidebar:
    st.header("👤 Your Profile")
    
    with st.expander("Set Up Personality Profile", expanded=st.session_state.personality_profile is None):
        st.markdown("Help us give you personalized advice!")
        
        age = st.selectbox(
            "Age Group",
            ["Under 13", "13-17", "18-24", "25-34", "35+"],
            help="We'll filter content appropriately"
        )
        
        purpose = st.selectbox(
            "Why do you use social media?",
            ["Just browsing/scrolling", "Engage in discussions", "Research/Learning", "Debate/Argue"],
            help="Helps us understand your goals"
        )
        
        comfort = st.selectbox(
            "Language comfort level",
            ["Prefer clean language", "Some casual language okay", "I can handle anything"],
            help="We'll warn you about content that exceeds this"
        )
        
        platform = st.multiselect(
            "Platforms you use",
            ["Reddit", "Twitter/X", "Facebook", "Instagram", "TikTok", "Other"],
            help="Optional: helps tailor advice"
        )
        
        if st.button("Save Profile", type="primary"):
            st.session_state.personality_profile = {
                'age': age,
                'purpose': purpose,
                'comfort_level': comfort,
                'platforms': platform
            }
            st.success("✅ Profile saved!")
            st.rerun()
    
    if st.session_state.personality_profile:
        st.success("✅ Profile Active")
        profile = st.session_state.personality_profile
        st.caption(f"Age: {profile['age']}")
        st.caption(f"Purpose: {profile['purpose']}")
        st.caption(f"Comfort: {profile['comfort_level']}")
        
        if st.button("Clear Profile"):
            st.session_state.personality_profile = None
            st.rerun()
    
    st.markdown("---")
    
    # Model info
    model_data = load_model()
    if model_data:
        _, meta = model_data
        if meta:
            st.subheader("🤖 Model Stats")
            st.metric("F1 Score", f"{meta['metrics']['f1']:.3f}")
            st.metric("AUC", f"{meta['metrics']['auc']:.3f}")

# Main app
st.title("🔥 ThreadCast: Smart Argument Predictor")
st.markdown("**AI-powered toxicity detection with personalized recommendations**")

# Check model
model_data = load_model()
if model_data is None:
    st.error("⚠️ Model not found! Run: `python3 training/train.py`")
    st.stop()

model_dict, meta = model_data

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Analyze Thread", "📊 Examples", "ℹ️ About"])

with tab1:
    st.markdown("### Paste Your Thread")
    st.markdown("Format: One message per line (optionally with `Username: message`)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        load_sample = st.selectbox("Quick Load", ["Custom"] + list(SAMPLE_THREADS.keys()))
    
    if load_sample != "Custom":
        default_text = SAMPLE_THREADS[load_sample]
    else:
        default_text = ""
    
    thread_text = st.text_area(
        "Thread Text",
        value=default_text,
        height=200,
        placeholder="User1: I disagree with this\nUser2: You're wrong about that\n...",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("🔍 Analyze Thread", type="primary", use_container_width=True)
    
    if analyze_btn and thread_text:
        with st.spinner("🧠 Analyzing thread..."):
            # Parse messages
            messages = parse_thread(thread_text)
            
            if len(messages) == 0:
                st.warning("No messages found. Please check your format.")
                st.stop()
            
            # Score messages
            scores = score_messages(messages, model_dict)
            
            for i, msg in enumerate(messages):
                msg['score'] = scores[i]
                msg['index'] = i + 1
            
            # Compute features
            features = compute_escalation_features(scores)
            risk_score = calculate_escalation_risk(features)
            
            # Detect triggers
            triggers = detect_trigger_messages(scores, threshold=0.2)
            
            # Forecast
            forecast = forecast_next_messages(scores, n_forecast=3)
            
            # Get personalized recommendation
            recommendations = get_recommendation(risk_score, st.session_state.personality_profile)
            
            # Display results
            st.markdown("---")
            
            # Personalized Recommendation Banner
            if risk_score > 70:
                banner_color = "error"
            elif risk_score > 40:
                banner_color = "warning"
            else:
                banner_color = "success"
            
            if banner_color == "error":
                st.error("🚨 **HIGH RISK THREAD**")
            elif banner_color == "warning":
                st.warning("⚠️ **MODERATE RISK THREAD**")
            else:
                st.success("✅ **SAFE THREAD**")
            
            # Show recommendations
            st.subheader("🎯 Personalized Recommendation")
            for rec in recommendations:
                st.markdown(rec)
            
            st.markdown("---")
            st.header("📊 Detailed Analysis")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_color = "🔴" if risk_score > 70 else "🟡" if risk_score > 40 else "🟢"
                st.metric("🔥 Escalation Risk", f"{risk_score:.0f}%", delta=risk_color)
            
            with col2:
                st.metric("📈 Trend", 
                         "↗️ Rising" if features['trend_slope'] > 0.01 else "→ Stable",
                         delta=f"{features['trend_slope']:.3f}")
            
            with col3:
                st.metric("💣 Triggers Found", len(triggers))
            
            with col4:
                st.metric("⚠️ High Toxicity", 
                         f"{features['high_toxicity_share']*100:.0f}%")
            
            # Charts
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Toxicity Timeline")
                
                df_plot = pd.DataFrame({
                    'Message': range(1, len(scores) + 1),
                    'Toxicity': scores
                })
                
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=df_plot['Message'],
                    y=df_plot['Toxicity'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))
                
                # Forecast
                forecast_x = range(len(scores) + 1, len(scores) + len(forecast) + 1)
                fig.add_trace(go.Scatter(
                    x=list(forecast_x),
                    y=forecast,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                fig.update_layout(
                    xaxis_title="Message #",
                    yaxis_title="Toxicity Score",
                    yaxis_range=[0, 1],
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribution")
                
                fig2 = go.Figure(data=[go.Histogram(
                    x=scores,
                    nbinsx=20,
                    marker_color='red',
                    opacity=0.7
                )])
                
                fig2.update_layout(
                    xaxis_title="Toxicity Score",
                    yaxis_title="Count",
                    xaxis_range=[0, 1],
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Trigger messages
            if triggers:
                st.markdown("---")
                st.subheader("💣 Trigger Messages")
                st.markdown("Messages that caused the biggest escalation jumps:")
                
                for idx, jump in triggers[:3]:
                    msg = messages[idx]
                    st.warning(f"**Message #{idx}** (Jump: +{jump:.2f})")
                    st.markdown(f"*{msg['user']}*: {msg['text']}")
                    st.progress(msg['score'])
                    st.caption(f"Toxicity: {msg['score']:.1%}")
            
            # All messages table
            st.markdown("---")
            st.subheader("📝 All Messages")
            
            df_messages = pd.DataFrame(messages)
            df_messages['toxicity'] = df_messages['score'].apply(lambda x: f"{x:.1%}")
            df_messages = df_messages[['index', 'user', 'text', 'toxicity']]
            
            st.dataframe(df_messages, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### 📊 Example Threads")
    st.markdown("See how different conversation styles are analyzed")
    
    for name, content in SAMPLE_THREADS.items():
        with st.expander(name):
            st.code(content)

with tab3:
    st.markdown("### ℹ️ About ThreadCast")
    
    st.markdown("""
    **ThreadCast** uses AI to predict when online conversations are about to explode into toxic arguments.
    
    #### 🎯 Features
    - **Toxicity Detection**: Analyzes each message for toxic language
    - **Escalation Prediction**: Forecasts how the conversation will evolve
    - **Trigger Identification**: Points out which messages sparked escalation
    - **Personalized Recommendations**: Tailored advice based on your profile
    
    #### 🤖 How It Works
    1. **TF-IDF Vectorization**: Converts text to numerical features
    2. **Logistic Regression**: Classifies toxicity (F1=0.673, AUC=0.959)
    3. **Trend Analysis**: Detects escalation patterns
    4. **Forecasting**: Predicts next 3 messages' toxicity
    
    #### 🎓 Use Cases
    - **Content Moderation**: Flag toxic threads before they escalate
    - **User Protection**: Warn users about harmful content
    - **Mental Health**: Help people avoid draining arguments
    - **Research**: Study online behavior patterns
    
    #### 🏆 Built For
    Silly-con Hackathon 2026
    
    ---
    
    **Tech Stack**: Python • Streamlit • scikit-learn • Plotly
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔥 ThreadCast • Protecting your peace online, one thread at a time</p>
    </div>
    """,
    unsafe_allow_html=True
)
