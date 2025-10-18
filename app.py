import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import numpy as np


st.set_page_config(
    page_title="The Language of Rejection",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    h1 {
        color: #2c3e50;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #3498db;
    }
    
    h2 {
        color: #34495e;
        font-weight: 600;
        margin-top: 30px;
    }
    
    h3 {
        color: #7f8c8d;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    
    .story-section {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .quote-box {
        background-color: #eef2f7;
        border-left: 4px solid #3498db;
        padding: 15px 20px;
        margin: 20px 0;
        font-style: italic;
        border-radius: 4px;
    }
    
    .highlight {
        background-color: #fff9c4;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    .danger-box {
        background-color: #ffebee;
        border-left: 4px solid #e74c3c;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2ecc71;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/rejection_analysis_extended.csv')
    df_clean = df[df['status'] != 'ghosted'].copy()
    
    with open('data/shap_results_all.json', 'r') as f:
        shap_data = json.load(f)
    
    return df_clean, shap_data

df, shap_results = load_data()
vader = SentimentIntensityAnalyzer()


@st.cache_data
def process_all_shap():
    """Extract all words with their impacts across companies"""
    all_negative = []
    all_positive = []
    
    for company, data in shap_results.items():
        for word, score in data['words']:
            if score < 0:
                all_negative.append({
                    'word': word,
                    'score': score,
                    'company': company,
                    'vader': data['vader'],
                    'roberta': data['roberta']
                })
            else:
                all_positive.append({
                    'word': word,
                    'score': score,
                    'company': company,
                    'vader': data['vader'],
                    'roberta': data['roberta']
                })
    
    df_neg = pd.DataFrame(all_negative)
    df_pos = pd.DataFrame(all_positive)
    
    return df_neg, df_pos

df_negative, df_positive = process_all_shap()


def analyze_text(text):
    """Analyze email text"""
    score = vader.polarity_scores(text)['compound']
    words = text.lower().split()
    
    joy_keywords = ['hope', 'happy', 'good', 'luck', 'best', 'wish', 'encourage']
    apology_keywords = ['sorry', 'unfortunately', 'regret', 'apologies', 'apologize']
    positive_keywords = ['thank', 'appreciate', 'grateful', 'impressed', 'value', 
                        'strong', 'excellent', 'great', 'pleased', 'interested']
    
    joy_count = sum(1 for w in words if any(j in w for j in joy_keywords))
    apology_count = sum(1 for w in words if any(a in w for a in apology_keywords))
    positive_count = sum(1 for w in words if any(p in w for p in positive_keywords))
    
    return {
        'score': score,
        'joy': joy_count,
        'apology': apology_count,
        'positive': positive_count,
        'word_count': len(words)
    }


st.sidebar.title("💔 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 The Story", "📊 The Data", "🔬 Deep Dive: SHAP", "🧪 Try It Yourself"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Emails Analyzed", "14")
st.sidebar.metric("Joy Factor", "+0.605")
st.sidebar.metric("The Magic Ratio", "4:1")

st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub](https://github.com/jgchoti/job_rejection_analysis) | [LinkedIn](https://www.linkedin.com/in/chotirat/)")

if page == "🏠 The Story":
    st.title("💔 The Language of Rejection")
    st.markdown("### What makes some rejection emails feel warm while others feel crushing?")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        st.image("assets/ophelia.jpg")
    
    # Introduction
    st.markdown("""
    <div class="story-section">
    <h2>📖 How This Started</h2>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    After receiving my 14th job rejection, something clicked. Instead of feeling frustrated, 
    I got curious: <b>could I quantify what makes a rejection feel "warm" or "cold"?</b>
    </p>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    I collected all 14 rejection emails, anonymized them, and analyzed them using 
    Natural Language Processing. What I found surprised me.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # The Dramatic Comparison
    st.markdown("### 📧 The Tale of Two Emails")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #e53935;">
        <h4 style="color: #c62828; margin-top: 0;">❌ The Coldest Email (0.307)</h4>
        <ul style="font-size: 1rem; line-height: 1.6;">
            <li>20 words total</li>
            <li>2 apologies</li>
            <li>2 positive words</li>
            <li>0 joy words</li>
        </ul>
        <p style="font-style: italic; color: #666; margin-bottom: 0;">
        "Thanks... Unfortunately... We're sorry..."
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #43a047;">
        <h4 style="color: #2e7d32; margin-top: 0;">✅ The Warmest Email (0.990)</h4>
        <ul style="font-size: 1rem; line-height: 1.6;">
            <li>88 words total</li>
            <li>0 apologies</li>
            <li>17 positive words</li>
            <li>2 joy words</li>
        </ul>
        <p style="font-style: italic; color: #666; margin-bottom: 0;">
        "We were impressed... We wish you the best..."
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quote-box">
    <b>The difference?</b> 68 words.
    Completely different emotional impact.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Findings
    st.markdown("### 🔍 What I Discovered")
    
    finding_col1, finding_col2, finding_col3 = st.columns(3)
    
    with finding_col1:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 100%;">
        <h3 style="color: #3498db; margin-top: 0;">🎉 Finding #1</h3>
        <h4>Joy Words Win</h4>
        <p style="font-size: 0.95rem; line-height: 1.6;">
        Words like "hope," "wish," and "best" correlate <b>+0.605</b> with warmth.
        That's 3× stronger than any other factor.
        </p>
        <p style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 0;">
        <i>The warmest emails average 2 joy words. The coldest? Zero.</i>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with finding_col2:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 100%;">
        <h3 style="color: #e74c3c; margin-top: 0;">⚖️ Finding #2</h3>
        <h4>The 4:1 Rule</h4>
        <p style="font-size: 0.95rem; line-height: 1.6;">
        Each apology ("sorry," "unfortunately") needs <b>at least 4 positive words</b> 
        to maintain warmth.
        </p>
        <p style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 0;">
        <i>Below 4:1 = guaranteed cold. Above 6:1 = safe zone.</i>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with finding_col3:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 100%;">
        <h3 style="color: #9b59b6; margin-top: 0;">🤖 Finding #3</h3>
        <h4>Context Matters</h4>
        <p style="font-size: 0.95rem; line-height: 1.6;">
        Transformers detect <b>hollow politeness</b> that simple word-counting misses.
        </p>
        <p style="font-size: 0.9rem; color: #7f8c8d; margin-bottom: 0;">
        <i>Some emails score 0.97 (lexicon) but -0.17 (transformer).</i>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Why It Matters
    st.markdown("""
    <div class="story-section">
    <h2>💡 Why This Matters</h2>
    <div style="font-size: 1.05rem; line-height: 1.8;">
    <p><b>For job seekers like me:</b> That sinking feeling after a cold rejection? 
    It's not personal. It's just a bad template. Some companies accidentally crush spirits. 
    Others lift them. Neither is intentional.</p>
    
    <p><b>For recruiters:</b> You can say "no" and still make someone's day better. 
    Small template improvements create measurably better candidate experiences. 
    Same automation. Different outcomes.</p>
    
    <p style="margin-bottom: 0;"><b>The bigger point:</b> Rejection is unavoidable in hiring. 
    Cruelty is optional.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("👉 **Explore the data in the next section to see exactly how we measured this.**")

elif page == "📊 The Data":
    st.title("📊 The Data Behind the Story")
    st.markdown("### Let's look at the numbers that prove these patterns are real")
    
    st.markdown("---")
    
    # Overview metrics
    st.markdown("## 📈 Dataset Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Emails", "14", help="Real rejection emails from 2024-2025 job search")
    with metric_col2:
        st.metric("Warmest Score", "0.990", delta="+0.68 vs coldest")
    with metric_col3:
        st.metric("Coldest Score", "0.307", delta="-0.68 vs warmest", delta_color="inverse")
    with metric_col4:
        st.metric("Average Score", f"{df['vader_compound'].mean():.3f}")
    
    st.markdown("---")
    
    # Finding #1: Joy Factor
    st.markdown("## 🎉 Finding #1: The Joy Factor")
    st.markdown("""
    **The Question:** Which emotions predict warmth best?
    
    **The Answer:** Joy-based words correlate **+0.605** with warmth. That's:
    - 5× stronger than trust (+0.116)
    - 10× stronger than anticipation (+0.057)
    """)
    
    # Graph 1: Joy vs Warmth
    fig1 = px.scatter(
        df,
        x='emotion_joy',
        y='vader_compound',
        size='afinn_positive_count',
        color='vader_compound',
        hover_data=['company_id', 'apology_words'],
        title="Joy Words Strongly Predict Warmth",
        labels={
            'emotion_joy': 'Number of Joy Words (hope, wish, best, good luck, etc.)',
            'vader_compound': 'Warmth Score',
            'afinn_positive_count': 'Total Positive Words'
        },
        color_continuous_scale='RdYlGn',
        height=500
    )
    fig1.update_traces(marker=dict(line=dict(width=1.5, color='white')))
    fig1.update_layout(
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.info("**💡 Key Insight:** Emails with 2+ joy words score 0.95+. Emails with 0 joy words score below 0.50. The pattern is clear and consistent.")
    
    # Graph 2: All emotions
    st.markdown("### 📊 All Emotions Ranked by Impact")
    
    emotion_cols = ['emotion_joy', 'emotion_trust', 'emotion_anticipation', 
                   'emotion_sadness', 'emotion_fear', 'emotion_anger']
    emotion_corr = df[emotion_cols].corrwith(df['vader_compound']).sort_values(ascending=False)
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=emotion_corr.index,
            y=emotion_corr.values,
            marker_color=['#2ecc71' if v > 0.2 else '#95a5a6' for v in emotion_corr.values],
            text=[f'{v:+.3f}' for v in emotion_corr.values],
            textposition='outside'
        )
    ])
    fig2.update_layout(
        title="Correlation of Each Emotion with Warmth Score",
        xaxis_title="Emotion Type",
        yaxis_title="Correlation with Warmth",
        template='plotly_white',
        height=400,
        xaxis={'tickangle': -45},
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Finding #2: The 4:1 Rule
    st.markdown("## ⚖️ Finding #2: The 4:1 Compensation Rule")
    st.markdown("""
    **The Question:** How many positive words are needed to balance one apology?
    
    **The Answer:** At least **4 positive words per apology**. Here's how we proved it:
    """)
    
    # Show the zones
    df_with_ratio = df[df['apology_words'] > 0].copy()
    df_with_ratio['ratio'] = df_with_ratio['afinn_positive_count'] / df_with_ratio['apology_words']
    df_with_ratio['zone'] = pd.cut(df_with_ratio['ratio'], 
                                    bins=[0, 4, 6, 20], 
                                    labels=['❌ Danger (<4:1)', '⚠️ Minimum (4-6:1)', '✅ Safe (6:1+)'])
    
    fig3 = px.scatter(
        df_with_ratio,
        x='ratio',
        y='vader_compound',
        color='zone',
        text='company_id',
        title="The 4:1 Rule: Positive-to-Apology Ratio vs Warmth",
        labels={
            'ratio': 'Positives per Apology',
            'vader_compound': 'Warmth Score'
        },
        color_discrete_map={
            '❌ Danger (<4:1)': '#e74c3c',
            '⚠️ Minimum (4-6:1)': '#f39c12',
            '✅ Safe (6:1+)': '#2ecc71'
        },
        height=500
    )
    fig3.update_traces(textposition='top center', marker=dict(size=15, line=dict(width=2, color='white')))
    fig3.add_vline(x=4, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Critical threshold (4:1)")
    fig3.add_vline(x=6, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="Safe zone starts (6:1)")
    fig3.add_hline(y=0.85, line_dash="dash", line_color="gray",
                  annotation_text="Warm threshold (0.85)")
    fig3.update_layout(template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Zone statistics
    zone_stats = df_with_ratio.groupby('zone').agg({
        'vader_compound': ['mean', 'count'],
        'company_id': 'count'
    }).round(3)
    
    st.markdown("### 📊 Success Rate by Zone")
    col1, col2, col3 = st.columns(3)
    
    danger_zone = df_with_ratio[df_with_ratio['zone'] == '❌ Danger (<4:1)']
    minimum_zone = df_with_ratio[df_with_ratio['zone'] == '⚠️ Minimum (4-6:1)']
    safe_zone = df_with_ratio[df_with_ratio['zone'] == '✅ Safe (6:1+)']
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 20px; border-radius: 10px;">
        <h4 style="color: #c62828;">❌ Danger Zone</h4>
        <p style="font-size: 1.8rem; font-weight: bold; margin: 10px 0;">0%</p>
        <p style="margin-bottom: 0;">None are warm (0.85+)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        warm_min = (minimum_zone['vader_compound'] >= 0.85).sum()
        total_min = len(minimum_zone)
        pct_min = (warm_min / total_min * 100) if total_min > 0 else 0
        st.markdown(f"""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px;">
        <h4 style="color: #f57c00;">⚠️ Minimum Zone</h4>
        <p style="font-size: 1.8rem; font-weight: bold; margin: 10px 0;">{pct_min:.0f}%</p>
        <p style="margin-bottom: 0;">{warm_min} of {total_min} are warm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        warm_safe = (safe_zone['vader_compound'] >= 0.85).sum()
        total_safe = len(safe_zone)
        pct_safe = (warm_safe / total_safe * 100) if total_safe > 0 else 0
        st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px;">
        <h4 style="color: #2e7d32;">✅ Safe Zone</h4>
        <p style="font-size: 1.8rem; font-weight: bold; margin: 10px 0;">{pct_safe:.0f}%</p>
        <p style="margin-bottom: 0;">{warm_safe} of {total_safe} are warm</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.success("**💡 Key Insight:** Below 4:1 ratio = 0% success rate. Above 4:1 = guaranteed warm. The threshold is empirically proven.")
    
    st.markdown("---")
    
    # All companies ranked
    st.markdown("## 📊 All 14 Companies Ranked")
    st.markdown("From warmest to coldest rejection emails:")
    
    df_sorted = df.sort_values('vader_compound', ascending=False)
    
    fig4 = go.Figure()
    colors = ['#2ecc71' if score >= 0.95 else '#3498db' if score >= 0.85 else '#f39c12' if score >= 0.60 else '#e74c3c' 
              for score in df_sorted['vader_compound']]
    
    fig4.add_trace(go.Bar(
        x=df_sorted['company_id'],
        y=df_sorted['vader_compound'],
        marker_color=colors,
        text=[f'{v:.3f}' for v in df_sorted['vader_compound']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br><extra></extra>'
    ))
    
    fig4.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="Very Warm (0.95+)")
    fig4.add_hline(y=0.85, line_dash="dash", line_color="orange", annotation_text="Warm (0.85+)")
    fig4.add_hline(y=0.60, line_dash="dash", line_color="gray", annotation_text="Neutral (0.60+)")
    
    fig4.update_layout(
        title="Company Warmth Rankings",
        xaxis_title="Company",
        yaxis_title="Warmth Score (VADER)",
        template='plotly_white',
        height=500,
        showlegend=False,
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Summary table
    st.markdown("### 📋 Detailed Breakdown")
    summary_df = df_sorted[['company_id', 'vader_compound', 'emotion_joy', 'afinn_positive_count', 'apology_words']].copy()
    summary_df.columns = ['Company', 'Warmth Score', 'Joy Words', 'Positive Words', 'Apologies']
    summary_df = summary_df.round(3)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


elif page == "🔬 Deep Dive: SHAP":
    st.title("🔬 Deep Dive: SHAP Word Analysis")
    st.markdown("### Which specific words push sentiment up or down? Let's find out.")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** reveals the exact contribution of each word 
    to RoBERTa's sentiment decision. This is where we find the **real insights** - 
    which words companies should use, and which they should avoid.
    """)
    
    st.markdown("---")
    
    # Tab structure for different views
    shap_tab1, shap_tab2, shap_tab3, shap_tab4 = st.tabs([
        "☠️ Words to Avoid",
        "⭐ Words to Use", 
        "🔄 Substitution Guide",
        "📚 Case Studies"
    ])
    

    with shap_tab1:
        st.markdown("## ☠️ The Most Damaging Words")
        st.markdown("These words carry the most negative weight across all emails:")
        
        # Get top 15 most negative
        top_negative = df_negative.nsmallest(15, 'score')
        
        fig_neg = go.Figure(data=[
            go.Bar(
                x=top_negative['score'],
                y=[f"{row['word']} ({row['company']})" for _, row in top_negative.iterrows()],
                orientation='h',
                marker_color='#e74c3c',
                text=[f"{score:.3f}" for score in top_negative['score']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
            )
        ])
        
        fig_neg.update_layout(
            title="Top 15 Most Damaging Words (Across All Companies)",
            xaxis_title="SHAP Value (Negative Impact)",
            yaxis_title="Word (Company)",
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_neg, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="danger-box">
            <h4>🚨 The "Unfortunately" Disaster</h4>
            <p><b>Company_C: -0.7727</b> (worst word in dataset!)</p>
            <p>This single word crushes sentiment. VADER thinks the email is positive (0.974), 
            but RoBERTa knows it's negative (-0.168) because transformers have learned 
            "unfortunately" is a rejection pattern.</p>
            <p style="margin-bottom: 0;"><b>Impact:</b> Using "unfortunately" creates a 1.14 point 
            disagreement between lexicons and transformers.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="danger-box">
            <h4>💀 The "Regret" Problem</h4>
            <p><b>Company_G: -0.7341</b> (second-worst!)</p>
            <p>"Regret" sounds polite, but it's actually <b>worse than "sorry"</b> (-0.47) 
            because it's strongly associated with formal rejections.</p>
            <p style="margin-bottom: 0;"><b>Advice:</b> Never use "regret" in rejection emails. 
            It's a learned rejection signal.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### 📊 Word Frequency Analysis")
        
        # Count how many times each negative word appears
        neg_word_counts = df_negative.groupby('word').agg({
            'score': ['mean', 'count', 'min']
        }).round(3)
        neg_word_counts.columns = ['avg_impact', 'frequency', 'worst_impact']
        neg_word_counts = neg_word_counts.sort_values('avg_impact').head(10)
        
        st.dataframe(
            neg_word_counts.reset_index().rename(columns={
                'word': 'Word',
                'avg_impact': 'Average Impact',
                'frequency': 'Appears in # Emails',
                'worst_impact': 'Worst Single Impact'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.warning("""
        **Bottom Line:** 
        - "Unfortunately" appears in 8 emails with average impact -0.47
        - "Sorry" appears in 4 emails with average impact -0.29
        - "Regret" appears in 2 emails but has devastating impact (-0.66 average)
        
        **Recommendation:** Avoid all three. If you must acknowledge disappointment, use "disappointing" (-0.17) instead.
        """)
    

    with shap_tab2:
        st.markdown("## ⭐ The Most Helpful Words")
        st.markdown("These words consistently boost warmth:")
        
        # Get top 15 most positive
        top_positive = df_positive.nlargest(15, 'score')
        
        fig_pos = go.Figure(data=[
            go.Bar(
                x=top_positive['score'],
                y=[f"{row['word']} ({row['company']})" for _, row in top_positive.iterrows()],
                orientation='h',
                marker_color='#2ecc71',
                text=[f"+{score:.3f}" for score in top_positive['score']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
            )
        ])
        
        fig_pos.update_layout(
            title="Top 15 Most Helpful Words (Across All Companies)",
            xaxis_title="SHAP Value (Positive Impact)",
            yaxis_title="Word (Company)",
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_pos, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 The Power Trio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>🏆 #1: "Appreciate"</h4>
            <p><b>Average Impact: +0.52</b></p>
            <p>Appears in 5 emails with consistently strong positive impact.</p>
            <p><b>Best Use:</b> Company_M (+0.59)</p>
            <p style="margin-bottom: 0;"><i>"We appreciate your interest..."</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>🥈 #2: "Thank"</h4>
            <p><b>Average Impact: +0.47</b></p>
            <p>Most frequent positive word. Appears in 10+ emails.</p>
            <p><b>Best Use:</b> Company_E (+0.56)</p>
            <p style="margin-bottom: 0;"><i>"Thank you for your time..."</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="success-box">
            <h4>🥉 #3: "Value"</h4>
            <p><b>Average Impact: +0.39</b></p>
            <p>Rare but powerful. Shows genuine appreciation.</p>
            <p><b>Best Use:</b> Company_F (+0.39)</p>
            <p style="margin-bottom: 0;"><i>"We value your effort..."</i></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Word frequency
        st.markdown("### 📊 Positive Word Frequency")
        
        pos_word_counts = df_positive.groupby('word').agg({
            'score': ['mean', 'count', 'max']
        }).round(3)
        pos_word_counts.columns = ['avg_impact', 'frequency', 'best_impact']
        pos_word_counts = pos_word_counts.sort_values('avg_impact', ascending=False).head(10)
        
        st.dataframe(
            pos_word_counts.reset_index().rename(columns={
                'word': 'Word',
                'avg_impact': 'Average Impact',
                'frequency': 'Appears in # Emails',
                'best_impact': 'Best Single Impact'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.success("""
        **Action Items:**
        - Use "appreciate" instead of just "thank" for stronger impact
        - Include "value" when you mean it (don't overuse)
        - "Impressed" (+0.51 peak) is powerful when specific
        - "Happy" and "wish" are joy words that boost warmth significantly
        """)
    

    with shap_tab3:
        st.markdown("## 🔄 Word Substitution Guide")
        st.markdown("### Simple swaps that make emails warmer:")
        
        # Create substitution table
        substitutions = [
            {
                'avoid': 'Sorry',
                'avoid_impact': -0.47,
                'use': 'Disappointed',
                'use_impact': -0.17,
                'improvement': '+64%',
                'example': 'We\'re disappointed we can\'t move forward'
            },
            {
                'avoid': 'Regret',
                'avoid_impact': -0.73,
                'use': 'Disappointing',
                'use_impact': -0.17,
                'improvement': '+77%',
                'example': 'This is disappointing news to share'
            },
            {
                'avoid': 'Unfortunately',
                'avoid_impact': -0.77,
                'use': '(Skip it!)',
                'use_impact': 0.0,
                'improvement': '+100%',
                'example': 'We\'ve decided to move forward with another candidate'
            },
            {
                'avoid': 'Thanks (generic)',
                'avoid_impact': -0.16,
                'use': 'Thank you + specific',
                'use_impact': +0.42,
                'improvement': '+163%',
                'example': 'Thank you for your thoughtful approach to the case study'
            },
            {
                'avoid': 'Thank (alone)',
                'avoid_impact': +0.13,
                'use': 'Appreciate',
                'use_impact': +0.54,
                'improvement': '+315%',
                'example': 'We appreciate your interest in our team'
            },
            {
                'avoid': 'Good luck',
                'avoid_impact': +0.12,
                'use': 'Wish you the best',
                'use_impact': +0.23,
                'improvement': '+92%',
                'example': 'We wish you the best in your job search'
            }
        ]
        
        df_sub = pd.DataFrame(substitutions)
        
        # Display as formatted table
        st.markdown("### 🔄 Recommended Substitutions")
        
        for _, row in df_sub.iterrows():
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; text-align: center;">
                <h4 style="color: #c62828; margin: 0;">❌ Avoid</h4>
                <p style="font-size: 1.2rem; font-weight: bold; margin: 10px 0;">{row['avoid']}</p>
                <p style="margin: 0; color: #666;">Impact: {row['avoid_impact']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                use_color = "#e8f5e9" if row['use_impact'] >= 0 else "#fff3e0"
                text_color = "#2e7d32" if row['use_impact'] >= 0 else "#f57c00"
                st.markdown(f"""
                <div style="background-color: {use_color}; padding: 15px; border-radius: 8px; text-align: center;">
                <h4 style="color: {text_color}; margin: 0;">✅ Use</h4>
                <p style="font-size: 1.2rem; font-weight: bold; margin: 10px 0;">{row['use']}</p>
                <p style="margin: 0; color: #666;">Impact: {row['use_impact']:+.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                <p style="margin: 0; color: #666;"><b>Improvement:</b> {row['improvement']}</p>
                <p style="margin: 5px 0 0 0; font-style: italic;">"{row['example']}"</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Key takeaways
        st.info("""
        **💡 Key Takeaway:** Small word changes create massive improvements:
        - Dropping "unfortunately" entirely = +100% better
        - Swapping "sorry" for "disappointed" = +64% better  
        - Using "appreciate" instead of "thank" = +315% better
        
        These are ONE-TIME template fixes that improve every rejection forever.
        """)

    with shap_tab4:
        st.markdown("## 📚 Case Studies: Winners & Losers")
        st.markdown("### Let's examine why certain companies succeeded or failed:")
        
        # Case Study 1: Company F (Winner)
        st.markdown("### 🏆 Case Study #1: Company F - The Winner (0.990)")
        
        company_f = shap_results['Company_F']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ What They Did Right</h4>
            <ul>
                <li><b>Strong positives:</b> Used powerful words like "appreciate" (+0.54), "value" (+0.39), "happy" (+0.26)</li>
                <li><b>Smart apology choice:</b> Used "disappointing" (-0.17) instead of "sorry" (-0.47)</li>
                <li><b>Joy words:</b> Included "wish" and "best"</li>
                <li><b>No "unfortunately":</b> Avoided the worst word</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Positive Force", f"+{company_f['words'][0][1] + company_f['words'][1][1]:.2f}")
            st.metric("Negative Force", f"{company_f['words'][-1][1] + company_f['words'][-2][1]:.2f}")
        
        with col2:
            # SHAP for Company F
            words_f = company_f['words']
            top_pos_f = sorted([w for w in words_f if w[1] > 0], key=lambda x: x[1], reverse=True)[:7]
            top_neg_f = sorted([w for w in words_f if w[1] < 0], key=lambda x: x[1])[:3]
            
            combined_f = top_pos_f + top_neg_f
            combined_f.sort(key=lambda x: x[1])
            
            fig_f = go.Figure(data=[
                go.Bar(
                    y=[w[0] for w in combined_f],
                    x=[w[1] for w in combined_f],
                    orientation='h',
                    marker_color=['#2ecc71' if w[1] > 0 else '#e74c3c' for w in combined_f],
                    text=[f"{w[1]:+.2f}" for w in combined_f],
                    textposition='outside'
                )
            ])
            
            fig_f.update_layout(
                title="Company F: Word Impact",
                xaxis_title="SHAP Value",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_f, use_container_width=True)
        
        st.success(f"""
        **Result:** 
        - VADER: {company_f['vader']:.3f} (very warm)
        - RoBERTa: {company_f['roberta']:.3f} (very warm)
        - **Gap: {abs(company_f['vader'] - company_f['roberta']):.3f}** (both models agree!)
        
        **The strategy worked:** Positive flooding + smart apology choice + joy words = warmest email in dataset.
        """)
        
        st.markdown("---")
        
        # Case Study 2: Company C (Loser)
        st.markdown("### 💔 Case Study #2: Company C - The Disagreement (VADER 0.974, RoBERTa -0.168)")
        
        company_c = shap_results['Company_C']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="danger-box">
            <h4>❌ What Went Wrong</h4>
            <ul>
                <li><b>The killer word:</b> "Unfortunately" with -0.7727 impact (worst in dataset!)</li>
                <li><b>Weak positives:</b> Generic words like "thanks" and "good"</li>
                <li><b>No compensation:</b> Not enough strong positives to balance apology</li>
                <li><b>VADER fooled:</b> Counted positive words, missed rejection context</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Positive Force", f"+{sum([w[1] for w in company_c['words'] if w[1] > 0]):.2f}")
            st.metric("Negative Force", f"{sum([w[1] for w in company_c['words'] if w[1] < 0]):.2f}")
        
        with col2:
            # SHAP for Company C
            words_c = company_c['words']
            top_pos_c = sorted([w for w in words_c if w[1] > 0], key=lambda x: x[1], reverse=True)[:5]
            top_neg_c = sorted([w for w in words_c if w[1] < 0], key=lambda x: x[1])[:5]
            
            combined_c = top_pos_c + top_neg_c
            combined_c.sort(key=lambda x: x[1])
            
            fig_c = go.Figure(data=[
                go.Bar(
                    y=[w[0] for w in combined_c],
                    x=[w[1] for w in combined_c],
                    orientation='h',
                    marker_color=['#2ecc71' if w[1] > 0 else '#e74c3c' for w in combined_c],
                    text=[f"{w[1]:+.2f}" for w in combined_c],
                    textposition='outside'
                )
            ])
            
            fig_c.update_layout(
                title="Company C: Word Impact",
                xaxis_title="SHAP Value",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_c, use_container_width=True)
        
        st.error(f"""
        **Result:**
        - VADER: {company_c['vader']:.3f} (thinks it's warm!)
        - RoBERTa: {company_c['roberta']:.3f} (knows it's cold!)
        - **Gap: {abs(company_c['vader'] - company_c['roberta']):.3f}** (BIGGEST disagreement in dataset!)
        
        **The lesson:** One terrible word ("unfortunately" -0.77) can destroy an entire email. 
        Transformers learned this is a rejection pattern, while lexicons just count words.
        """)
        
        st.markdown("---")
        
        # Comparison table
        st.markdown("### 📊 Winner vs Loser Comparison")
        
        comparison = pd.DataFrame({
            'Metric': [
                'VADER Score',
                'RoBERTa Score',
                'Model Agreement',
                'Positive Word Sum',
                'Negative Word Sum',
                'Net Impact',
                'Joy Words Used',
                'Worst Word Impact',
                'Best Word Impact'
            ],
            'Company F (Winner)': [
                f"{company_f['vader']:.3f}",
                f"{company_f['roberta']:.3f}",
                f"{abs(company_f['vader'] - company_f['roberta']):.3f} ✅",
                "+2.67 🏆",
                "-0.45 ✅",
                "+2.22 🏆",
                "2 (wish, best)",
                "-0.17 (disappointing)",
                "+0.42 (thank)"
            ],
            'Company C (Loser)': [
                f"{company_c['vader']:.3f}",
                f"{company_c['roberta']:.3f}",
                f"{abs(company_c['vader'] - company_c['roberta']):.3f} ❌",
                "+1.03",
                "-1.59 ❌",
                "-0.56 ❌",
                "1 (good)",
                "-0.77 (unfortunately) ☠️",
                "+0.18 (thanks)"
            ]
        })
        
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **The Contrast:**
        - **Positive sum:** Winner has 2.6× more positive force
        - **Negative sum:** Winner has 72% less negative force
        - **Net impact:** Winner is +2.78 points better
        - **Model agreement:** Winner has models aligned, loser has 1.14 gap
        
        **Bottom line:** Company F's template is objectively, measurably better.
        """)

else:  # Try It Yourself
    st.title("🧪 Try It Yourself")
    st.markdown("### Analyze your own rejection email or create a better template")
    
    tab1, tab2 = st.tabs(["🔍 Analyze", "✍️ Rewrite"])
    
    with tab1:
        st.markdown("## 📧 Email Analyzer")
        st.markdown("Paste any rejection email to see how it scores:")
        
        # Example selector
        example = st.selectbox(
            "Or try an example:",
            ["", "❌ Cold Email", "😐 Generic Email", "✅ Warm Email"]
        )
        
        example_texts = {
            "❌ Cold Email": """Dear [Name],

Thanks for applying to [Role].

Unfortunately, we have decided not to move forward with your application at this time. We're sorry we couldn't offer you a position.

Thanks again,
[Team]""",
            "😐 Generic Email": """Hi [Name],

Thank you for your interest in the [Role] position.

After careful consideration, we have decided to move forward with other candidates whose qualifications more closely match our needs.

We wish you success in your job search.

Best regards,
[Recruiter]""",
            "✅ Warm Email": """Dear [Name],

Thank you for your interest in [Role] and for taking the time to interview with us.

We were impressed by your background in [field] and appreciated learning about your experience with [topic]. While we're moving forward with another candidate whose skills more closely align with this specific role, we want you to know this was a very competitive process.

We encourage you to watch for future opportunities that match your expertise.

We wish you the best in your search!

Warm regards,
[Name]"""
        }
        
        text_input = st.text_area(
            "Paste email here:",
            value=example_texts.get(example, ""),
            height=250,
            placeholder="Dear [Name],\n\nThank you for applying..."
        )
        
        if st.button("🚀 Analyze Email", type="primary"):
            if len(text_input.strip()) < 10:
                st.warning("Please paste an email first!")
            else:
                results = analyze_text(text_input)
                
                # Results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Gauge and metrics
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Simple gauge visualization
                    score = results['score']
                    if score >= 0.95:
                        st.success(f"### ✅ Very Warm\n**Score: {score:.3f}**")
                    elif score >= 0.85:
                        st.info(f"### 😊 Warm\n**Score: {score:.3f}**")
                    elif score >= 0.60:
                        st.warning(f"### 😐 Neutral\n**Score: {score:.3f}**")
                    else:
                        st.error(f"### ❌ Cold\n**Score: {score:.3f}**")
                
                with col2:
                    st.markdown("#### 📈 Breakdown")
                    st.metric("Joy Words", results['joy'])
                    st.metric("Positive Words", results['positive'])
                    st.metric("Apologies", results['apology'])
                
                # Analysis
                st.markdown("### 🔍 Analysis")
                
                problems = []
                if results['apology'] > 1:
                    problems.append(f"⚠️ Too many apologies ({results['apology']}). Limit to 0-1.")
                elif results['apology'] == 1 and results['positive'] < 4:
                    problems.append(f"⚠️ Not enough positives for apology. Need 4+, have {results['positive']}.")
                
                if results['joy'] == 0:
                    problems.append("⚠️ No joy words. Add 'wish', 'hope', or 'best'.")
                
                if results['positive'] < 4:
                    problems.append(f"⚠️ Only {results['positive']} positive words. Aim for 6+.")
                
                if problems:
                    st.error("**Problems Detected:**")
                    for p in problems:
                        st.markdown(f"- {p}")
                else:
                    st.success("✅ No major problems detected!")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if results['score'] >= 0.95:
                    st.success("This is an excellent rejection email! It balances warmth with professionalism.")
                elif results['score'] >= 0.85:
                    st.info("This is a warm email. Consider adding more joy words to make it exceptional.")
                else:
                    st.warning("""
                    This email could be warmer. Try:
                    - Adding joy words like "wish you the best"
                    - Increasing positive words to 6+
                    - Following the 4:1 rule (4 positives per apology)
                    - Avoiding "unfortunately" and "regret"
                    """)
    
    with tab2:
        st.markdown("## ✍️ Email Rewriter")
        st.markdown("Paste a cold email template and we'll suggest improvements:")
        
        rewrite_input = st.text_area(
            "Paste template here:",
            height=200,
            placeholder="Dear [Name],\n\nThanks for applying..."
        )
        
        if st.button("🔍 Analyze & Improve", type="primary"):
            if len(rewrite_input.strip()) < 10:
                st.warning("Please paste an email first!")
            else:
                original = analyze_text(rewrite_input)
                
                # Simple improvements
                improved = rewrite_input
                suggestions = []
                
                # Remove "unfortunately"
                if 'unfortunately' in improved.lower():
                    improved = re.sub(r'\b[Uu]nfortunately,?\s*', '', improved)
                    suggestions.append("✅ Removed 'unfortunately' (saves up to -0.77 negative impact!)")
                
                # Replace "sorry" with "disappointed"
                if 'sorry' in improved.lower():
                    improved = re.sub(r"\bsorry\b", "disappointed", improved, flags=re.IGNORECASE)
                    suggestions.append("✅ Changed 'sorry' to 'disappointed' (+64% less negative)")
                
                # Replace "regret" with "disappointing"
                if 'regret' in improved.lower():
                    improved = re.sub(r"\bregret\b", "find it disappointing", improved, flags=re.IGNORECASE)
                    suggestions.append("✅ Changed 'regret' to 'disappointing' (+77% less negative)")
                
                # Add joy words
                if original['joy'] == 0 and 'wish you' not in improved.lower():
                    improved += "\n\nWe wish you the best in your job search!"
                    suggestions.append("✅ Added joy words to closing (+0.23 impact)")
                
                # Add specific positive
                if original['positive'] < 4 and 'impressed' not in improved.lower() and 'appreciate' not in improved.lower():
                    lines = improved.split('\n')
                    for i, line in enumerate(lines):
                        if any(word in line.lower() for word in ['dear', 'hi ', 'hello']):
                            lines.insert(i+1, "\nWe appreciate the time you invested in your application and were impressed by your background.\n")
                            break
                    improved = '\n'.join(lines)
                    suggestions.append("✅ Added 'appreciate' and 'impressed' (+1.05 combined impact)")
                
                # Enhance "thank" to "appreciate"
                if improved.count('thank') > 0 and 'appreciate' not in improved.lower():
                    improved = improved.replace('Thank you', 'We appreciate your', 1)
                    suggestions.append("✅ Enhanced 'thank' to 'appreciate' (+300% more impact)")
                
                improved_analysis = analyze_text(improved)
                
                # Show results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Before", f"{original['score']:.3f}")
                with col2:
                    st.metric("After", f"{improved_analysis['score']:.3f}")
                with col3:
                    change = improved_analysis['score'] - original['score']
                    st.metric("Improvement", f"{change:+.3f}", delta=f"{change/abs(original['score'])*100:+.0f}%")
                
                st.markdown("### ✨ Changes Applied")
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"- {s}")
                else:
                    st.success("Email is already optimal!")
                
                st.markdown("### 📧 Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original:**")
                    st.text_area("", value=rewrite_input, height=300, disabled=True, label_visibility="collapsed")
                
                with col2:
                    st.markdown("**Improved:**")
                    st.text_area("", value=improved, height=300, disabled=True, label_visibility="collapsed")
                
                # Show the math
                if suggestions:
                    st.info(f"""
                    **💡 Why This Works:**
                    
                    Based on SHAP analysis of 14 real emails:
                    - Removing "unfortunately" = up to +0.77 improvement
                    - Swapping "sorry" for "disappointed" = +0.30 improvement
                    - Adding "appreciate" instead of "thank" = +0.40 improvement
                    - Adding joy words = +0.20+ improvement
                    
                    **Total potential improvement: {change:+.2f} points!**
                    """)

# Run
if __name__ == "__main__":
    pass