import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

st.set_page_config(
    page_title="The Language of Rejection",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers */
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
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }
    
    /* Buttons */
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
    
    /* Tabs */
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
    
    /* Story sections */
    .story-section {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Quote boxes */
    .quote-box {
        background-color: #eef2f7;
        border-left: 4px solid #3498db;
        padding: 15px 20px;
        margin: 20px 0;
        font-style: italic;
        border-radius: 4px;
    }
    
    /* Highlight */
    .highlight {
        background-color: #fff9c4;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('data/rejection_analysis_extended.csv')
    df_clean = df[df['status'] != 'ghosted'].copy()
    
    with open('data/shap_results.json', 'r') as f:
        shap_data = json.load(f)
    
    return df_clean, shap_data

df, shap_results = load_data()

vader = SentimentIntensityAnalyzer()


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
    ["🏠 The Story", "📊 The Data", "🔬 Deep Dive", "🧪 Try It Yourself"],
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
    
    # Introduction
    st.markdown("""
    <div class="story-section">
    <h2>📖 How This Started</h2>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    After receiving too many job rejection, something clicked. Instead of feeling frustrated, 
    I got curious: <b>could I quantify what makes a rejection feel "warm" or "cold"?</b>
    </p>
    <p style="font-size: 1.1rem; line-height: 1.8;">
    I collected all rejection emails, anonymized them, and analyzed them using 
    Natural Language Processing(NLP).
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
    <b>The difference?</b> 68 words more. 
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
        <h3 style="color: #9b59b6; margin-top: 0;">🧪 Finding #3</h3>
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

    <p><b>🔍 For job seekers:</b> That sinking feeling after a cold rejection? 
    It's rarely personal. Often, it's just a poorly worded template. Some companies unintentionally discourage candidates, while others leave a positive impression. Recognizing the patterns helps us navigate rejections with less stress.</p>

    <p><b>🏣 For recruiters:</b> Saying "no" can still be kind. Thoughtful templates, small personalized touches, or empathetic phrasing can significantly improve candidate experience. As AI-generated communications become standard, these small design choices will define company reputation and candidate satisfaction.</p>

    <p><b>🔮 Looking ahead:</b> In the future, AI may read rejection emails before humans, or even ranking or analyzing candidate responses. A “warm” message won't just feel better to the recipient; it could affect how automated systems evaluate engagement, sentiment, and candidate follow-up. Crafting messages with warmth and clarity becomes a strategic move for both human experience and AI-driven workflows.</p>

    <p style="margin-bottom: 0;"><b>✨ The bigger point:</b> Rejection is unavoidable in hiring but unnecessary cruelty is optional.</p>

    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("👉 **Explore the data in the next section to see exactly how I measured this.**")

# ============================================
# PAGE 2: THE DATA
# ============================================
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

# ============================================
# PAGE 3: DEEP DIVE
# ============================================
elif page == "🔬 Deep Dive":
    st.title("🔬 Deep Dive: SHAP Analysis")
    st.markdown("### Which specific words push sentiment up or down?")
    
    st.markdown("""
    SHAP (SHapley Additive exPlanations) reveals the **exact contribution** of each word 
    to the RoBERTa transformer's sentiment decision. This shows us WHY some emails feel warm or cold.
    """)
    
    st.markdown("---")
    
    # Company selector
    companies = sorted(shap_results.keys())
    selected_company = st.selectbox(
        "Select a company to analyze:",
        companies,
        format_func=lambda x: f"{x} (VADER: {shap_results[x]['vader']:.3f}, RoBERTa: {shap_results[x]['roberta']:.3f})"
    )
    
    if selected_company:
        data = shap_results[selected_company]
        words = data['words']
        
        # Get top words
        words_sorted = sorted(words, key=lambda x: x[1], reverse=True)
        top_positive = [w for w in words_sorted if w[1] > 0][:10]
        top_negative = sorted([w for w in words if w[1] < 0], key=lambda x: x[1])[:10]
        
        combined = top_positive + top_negative
        combined.sort(key=lambda x: x[1])
        
        words_list = [w[0] for w in combined]
        scores = [w[1] for w in combined]
        colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores]
        
        # Create visualization
        fig = go.Figure(data=[
            go.Bar(
                y=words_list,
                x=scores,
                orientation='h',
                marker_color=colors,
                text=[f"{s:+.3f}" for s in scores],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"{selected_company}: Word-Level Impact on Sentiment",
            xaxis_title="SHAP Value (Impact on Sentiment)",
            yaxis_title="Word",
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        fig.add_vline(x=0, line_width=2, line_color="gray", line_dash="solid")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        total_positive = sum([s for s in scores if s > 0])
        total_negative = sum([s for s in scores if s < 0])
        net = total_positive + total_negative
        gap = abs(data['vader'] - data['roberta'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Positive Force", f"+{total_positive:.3f}")
        with col2:
            st.metric("Negative Force", f"{total_negative:.3f}")
        with col3:
            st.metric("Net Impact", f"{net:+.3f}")
        with col4:
            st.metric("VADER-RoBERTa Gap", f"{gap:.3f}")
        
        # Explanation
        st.markdown("### 💡 What This Means")
        
        if gap > 0.5:
            st.warning(f"""
            **Large Disagreement ({gap:.3f}):** VADER and RoBERTa strongly disagree. 
            VADER scored this **{data['vader']:.3f}** ({"very positive" if data['vader'] > 0.8 else "positive"}) 
            while RoBERTa scored **{data['roberta']:.3f}** ({"negative" if data['roberta'] < 0 else "barely positive"}).
            
            **Why?** VADER counts positive words. RoBERTa understands that rejection phrases 
            like "unfortunately...not proceed" carry more negative weight than scattered positive words.
            """)
        else:
            st.success(f"""
            **Agreement ({gap:.3f}):** VADER and RoBERTa mostly agree. 
            Both scored this email as {"warm" if data['vader'] > 0.85 else "cold" if data['vader'] < 0.5 else "neutral"}.
            
            The balance of positive and negative words is clear enough that even simple word counting works well.
            """)

# ============================================
# PAGE 4: TRY IT YOURSELF
# ============================================
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
            ["", "❌ Cold Email", "😐 Generic Email", "✅ Warm Email"],
            label_visibility="collapsed"
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
                
                # Add joy words
                if original['joy'] == 0 and 'wish you' not in improved.lower():
                    improved += "\n\nWe wish you the best in your job search!"
                    suggestions.append("✅ Added joy words to closing")
                
                # Add specific positive
                if original['positive'] < 4 and 'impressed' not in improved.lower():
                    lines = improved.split('\n')
                    for i, line in enumerate(lines):
                        if any(word in line.lower() for word in ['dear', 'hi ', 'hello']):
                            lines.insert(i+1, "\nWe were impressed by your background and the time you invested in your application.\n")
                            break
                    improved = '\n'.join(lines)
                    suggestions.append("✅ Added specific positive feedback")
                
                # Remove excess apologies
                if original['apology'] > 1:
                    improved = re.sub(r',\s*unfortunately\b', '', improved, flags=re.IGNORECASE)
                    improved = re.sub(r'\bUnfortunately,\s*', '', improved, flags=re.IGNORECASE)
                    suggestions.append("✅ Removed excess apologies")
                
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

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    pass