# Silly-con-Hackathon-2026
Team name: Brain.exe-Crashed
Project: AI That Predicts Internet Arguments Before They Explode

Youtube Demo: https://www.youtube.com/watch?v=NuuJn88PzjY


# Inspiration
Online discussions frequently escalate from minor disagreements into personal attacks within a few replies. Most moderation systems detect toxicity only after harmful language has already appeared.

We were inspired by a different question:

What if we could predict when a conversation is about to escalate, rather than just labeling it as toxic afterward?

Our goal was to build a system that forecasts conversational escalation in real time, helping users avoid emotionally draining threads and supporting healthier online discourse.

## What it does
ThreadCast predicts whether a conversation is likely to escalate into a toxic argument.

Given a pasted conversation thread, the system:
	•	Assigns a toxicity score to each message
	•	Analyzes the trajectory of toxicity over time
	•	Computes an overall escalation risk score (0–100%)
	•	Identifies trigger messages that caused sharp increases
	•	Forecasts the toxicity of upcoming messages
	•	Provides personalized recommendations based on user preferences

Instead of static classification, our system models conversational dynamics.
## How we built it
We structured the system in three main components.

1. Toxicity Classification Layer

We trained a supervised text classification model using:
	•	TF-IDF vectorization
	•	Logistic Regression
	•	Public toxicity datasets (Reddit-style and Jigsaw-inspired data)

The model outputs a probability score (0–1) representing message-level toxicity.

⸻

2. Escalation Modeling Layer

On top of per-message toxicity, we engineered time-series features:
	•	Trend slope (toxicity growth rate)
	•	Volatility (standard deviation across recent messages)
	•	High-toxicity share
	•	Sudden jump detection (trigger analysis)
	•	Short-term linear forecasting
These features are combined into a weighted escalation risk score.

This transforms toxicity detection into a trajectory prediction problem.
3. Interactive Web Application

We built an interactive Streamlit web application that:
	•	Allows users to paste conversation threads
	•	Visualizes toxicity timelines using Plotly
	•	Highlights escalation triggers
	•	Displays risk metrics and forecasts
	•	Generates personalized recommendations

The system is lightweight, fast, and deployable.

4. Personalizations for all age groups and different personal uses

## Challenges we ran into
	1.	Reframing the problem
Most toxicity tools focus on classification. We had to redesign the system to model escalation as a temporal process.
	2.	Noisy real-world data
Online comments contain sarcasm, slang, misspellings, and context-dependent meaning. Data cleaning and preprocessing were critical.
	3.	Balancing performance and speed
Under hackathon constraints, we needed a model that was accurate but also fast to train and deploy. We prioritized a strong, interpretable baseline over large transformer models.
	4.	Forecasting uncertainty
Predicting future toxicity introduces uncertainty. We designed a simple yet stable short-term forecasting approach suitable for real-time use.


The system is lightweight, fast, and deployable.
## Accomplishments that we're proud of
	•	Built a complete ML pipeline from data preprocessing to deployment
	•	Achieved strong validation performance (F1 ≈ 0.67, AUC ≈ 0.95)
	•	Implemented escalation forecasting rather than static toxicity detection
	•	Developed a personalized recommendation layer
	•	Delivered a functional, demo-ready web application within hackathon constraints

Most importantly, we reframed toxicity detection as a preventative tool.

## What we learned
	•	Escalation patterns often appear gradually before extreme toxicity emerges
	•	Simpler models (TF-IDF + Logistic Regression) can perform competitively
	•	Time-series feature engineering significantly enhances interpretability
	•	UX design strongly affects how users perceive AI-driven risk signals
	•	End-to-end ML system building requires prioritization under time pressure

## What's next for AI That Predicts Internet Arguments Before They Explode
	1.	Context-aware transformer models
Fine-tune BERT-style models to incorporate multi-message context.
	2.	Real-time integration
Develop a browser extension that warns users before posting into escalating threads.
	3.	Network-level analysis
Model user interaction graphs to identify escalation catalysts.
	4.	Mental health integration
Provide de-escalation suggestions and well-being safeguards.
	5.	Platform moderation tools
Build dashboards for community moderators to proactively detect risky threads.

--With different socia media API's, one would be able to copy the URL links of threads and prevent their peace from being disturbed.
