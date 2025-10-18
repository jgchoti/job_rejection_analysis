# The Language of Rejection: NLP Analysis of Job Rejection Emails

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20RoBERTa%20%7C%20SHAP-orange)

> An exploratory analysis revealing how AI sentiment models interpret professional rejection communication - and what separates algorithmically "warm" rejections from genuinely empathetic ones.

---

## 🎯 TL;DR

**Key Findings:**

- 💬 **Joy words predict warmth**: "Hope," "wish," "best" correlate +0.605 with sentiment (3× stronger than any other factor)
- ⚖️ **The 4:1 rule**: Each apology needs 4+ positive words to maintain warmth
- 🤖 **Transformers > Lexicons**: Context-aware models detect patterns lexicons miss
- 📧 **Automation ≠ coldness**: Templates can be warm with the right wording

**Dataset:** Real rejection emails analyzed with VADER, RoBERTa, emotion lexicons, and SHAP.

⚠️ **Living project**: New rejection emails will be added over time. Findings may evolve as the dataset grows.

## 📂 Project Structure

- `data/`: Raw and processed rejection email data
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `scripts/`: Python scripts for data processing
- `visualizations/`: Charts and graphs generated from the analysis
- `README.md`: This overview and summary of findings
- `app.py`: interactive dashboard application

---

## 💡 Why I Did This

After receiving many job rejections, I noticed patterns: some emails felt respectful, others dismissive, some never replied at all. As someone studying data science, I wondered: **could I quantify what makes a rejection feel "warm" or "cold"?**

This project started as personal (processing rejection through data) and evolved into a technical exploration (revealing fundamental limitations in how AI interprets human communication).

---

## 📧 A Note on Automated Rejections

**Reality check:** Most rejection emails are automated or templated. And that's okay.

This analysis doesn't expect recruiters to hand-craft every rejection. At scale, that's impossible. However, **automation doesn't have to mean coldness.**

**Why This Matters**: Your emails aren't just being read by humans anymore. Understanding which phrases feel warm to both humans and AI can help craft communications that are effective, readable, and positively received.

---

## 📊 What I Discovered

### Finding #1: Joy Words Are Everything

![Joy vs Sentiment](visualizations/emotion_sentiment_heatmap.png)
_Figure 1: Strong correlation between joy-based emotion words and overall sentiment score._

**Discovery:** Joy-based emotion (NRC lexicon) is the strongest predictor of sentiment (r = +0.6). 5x stronger than trust (+0.12), 10x stronger than anticipation (+0.06)

**What are joy words?**

- "hope", "wish", "best", "luck", "happy", "good", "encourage"
- Measured using NRC Emotion Lexicon

---

### Finding #2: The Magic Number is 4

Rejection emails need **at least 4 positive words** to feel warm.

#### The Three Zones

| Zone             | Ratio | Success Rate  | Examples               |
| ---------------- | ----- | ------------- | ---------------------- |
| **Danger Zone**  | < 4:1 | 0% (all cold) | Company_D: 1:1 → 0.307 |
| **Minimum Zone** | 4-6:1 | 100% warm     | Company_E: 4:1 → 0.863 |
| **Safe Zone**    | 6:1+  | 100% warm     | Company_B: 8:1 → 0.988 |

![Positive Word Gradient](visualizations/4_to_1_threshold_analysis.png)

#### How I Calculated It

**Method 1: Linear Regression**

```
Sentiment = 0.825 + (0.025 × Positives) - (0.154 × Apologies)

Each apology: -0.154 impact
Each positive: +0.025 impact
Breakeven ratio: 0.154 ÷ 0.025 = 6.2:1
```

**Method 2: Empirical Threshold**

- Company_D at 1:1 → 0.307 (cold)
- Company_E at 4:1 → 0.863 (warm) ← **First warm email**
- All emails 6:1+ → 0.93+ (very warm)

**Method 3: Correlation Optimization**

- Tested ratios 1:1 through 10:1
- Peak correlation at 6:1 (r = 0.767)
- But warmth begins at 4:1

## **Conclusion:** 4:1 is the minimum, 6:1 is optimal.

### Finding #3: The Lexicon vs Transformer Gap

![compare_models.png](visualizations/correlation_heatmap_models.png)
_Figure 6: Correlation heatmap comparing lexicon-based models (VADER, TextBlob, AFINN) versus transformer models (RoBERTa, SST-2)._

**Discovery:** Lexicon-based models (VADER, AFINN) can be fooled by polite language, while transformer models (RoBERTa) detect the underlying rejection.

**The correlations:**

- **Within lexicons**: VADER ↔ TextBlob (0.73), VADER ↔ AFINN (0.59)
- **Within transformers**: RoBERTa ↔ SST-2 (0.48)
- **Across paradigms**: VADER ↔ RoBERTa (0.38) ⚠️ weak!

**Why lexicons fail:**

```
VADER logic:
  Count: 16 positive words, 4 negative words
  Math: 16 - 4 = +12
  Result: 0.988 (very positive!)
  Human perception: "This is clearly a rejection"
```

**Why transformers succeed:**

```
RoBERTa logic:
  Detects: "unfortunately... decided not to proceed"
  Understands: This is a rejection phrase
  Weights: Negative phrase > scattered positive words
  Result: +0.17 (barely positive)
  Human perception: "Polite but still a rejection"
```

---

### Finding #4: SHAP Reveals the Balance

**Discovery:** Using SHAP (SHapley Additive exPlanations), we can see exactly which words push sentiment up or down.

![SHAP Word Attributions](visualizations/shap_word_importance_4panel.png)
_Figure 3: Word-level importance analysis showing which words drive RoBERTa's sentiment decisions for each email._

#### Case Study: Company B (VADER 0.988, RoBERTa 0.170)

**Why the huge gap?**

**Positive forces (SHAP values):**

- "great" → +0.51
- "amongst" → +0.34
- "interest" → +0.12
- "best" → +0.10
- **Total positive: +2.46**

**Negative forces:**

- "unfortunately" → -0.38
- "proceed" → -0.35
- "not" → -0.13
- "sorry" → -0.08
- **Total negative: -1.30**

**Net impact: +2.46 - 1.30 = +1.16**

**The problem:** Despite having positive words, the phrase "unfortunately, we decided not to proceed" carries massive negative weight (-1.30 combined). Positive words BARELY overcome this.

#### Comparison: Company F (VADER 0.990, RoBERTa 0.874)

**Both agree it's warm. Why?**

**Positive forces:**

- "Thank" → +0.42
- "value" → +0.39
- "happy" → +0.26
- "impressed" → +0.18
- "effort" → +0.17
- **Total positive: +2.67**

**Negative forces:**

- "disappointing" → -0.17
- "unfortunately" → -0.12
- **Total negative: -0.45**

**Net impact: +2.67 - 0.45 = +2.22**

**The strategy:** Floods with strong positive words that OVERWHELM weak negatives. Uses "disappointing" (less negative) instead of "sorry" (more negative).

#### Comparison: Company D (Both Agree It's Cold)

**VADER 0.307, RoBERTa -0.282**

**Positive forces:**

- "Hopefully" → +0.12
- "skills" → +0.11
- **Total positive: +0.87**

**Negative forces:**

- "sorry" → -0.47
- "Unfortunately" → -0.42
- **Total negative: -1.41**

**Net impact: +0.87 - 1.41 = -0.54**

**The problem:** Strong negative words with minimal positive compensation. Both models agree: this is cold.

## 🔍 How I Did This

### The Data

- 14 real rejection emails from my 2024-2025 job search
- All company names removed for privacy
- Belgium-based tech/data job applications
- Anonymized (Company A through Company N)

### The Analysis

**Phase 1: Multi-Model Sentiment Analysis**

1. **Lexicon-based**: VADER, TextBlob, AFINN
   - Rule-based, count positive/negative words
   - Fast, interpretable, but context-blind
2. **Transformer-based**: RoBERTa (cardiffnlp), DistilBERT (SST-2)
   - Context-aware, phrase understanding
   - Slower but more nuanced

**Phase 2: Feature Engineering**

- Linguistic metrics: word count, sentence count, readability
- Keyword detection: empathy words, apology words, personal pronouns
- NRC Emotion Lexicon: joy, trust, anticipation, sadness, fear, anger
- AFINN positive/negative word counts
- Boolean features: mentions future, contains feedback

**Phase 3: Explainability Analysis**

- SHAP (SHapley Additive exPlanations) for word-level attributions
- transformers-interpret library for RoBERTa explainability
- Identified which exact words drive positive vs negative predictions

---

## 💡 What This Means

### For Job Seekers (That's Me!)

- **Don't take it personally** - "warmth" is often just following a template
- **Red flag**: Email with 0 joy words and 2+ apologies = they didn't try
- **Green flag**: 2+ joy words and personal details = they actually care

### For Companies Writing Rejections

**Want to make rejections feel warmer? Here's the cheat sheet:**

1. **Include joy words** (biggest impact):

   - "We wish you the best in your search"
   - "Good luck with your future endeavors"
   - "We hope you find a great fit"

2. **Follow the 4:1 rule**:

   - Each "sorry" or "unfortunately" needs 4+ positive words
   - Better: 1 apology + 6 positives
   - Best: 0 apologies + focus on positives

3. **Be specific**:
   - "We were impressed by your X" is better than "You're qualified"
   - "Thank you for your thoughtful approach to Y" is better than "Thanks for applying"

---

## 🛠️ Technical Details

### Tools

- Python for all analysis
- VADER, TextBlob, AFINN for basic sentiment
- RoBERTa transformer model for smart AI analysis
- NRC Emotion Lexicon for emotion detection
- SHAP for explaining which words matter
- Matplotlib/Seaborn for charts

---

## ⚠️ Limitations

**Small sample:** Only 13 emails. This is exploratory, not definitive.

**Paraphrased:** I rewrote emails to protect company privacy, which might have changed some details.

**Ongoing work**: As I continue collecting new job rejection emails, additional data will be incorporated to refine and validate these findings. Patterns may evolve with a larger sample, so insights here are subject to update.

## 🔮 Future Relevance

As AI increasingly becomes the “first reader” of written communications, humans will often rely on AI-generated summaries or AI-curated messages. Understanding which phrases convey warmth and positivity—both to humans and AI allows business to craft emails and content that are simultaneously empathetic, readable, and algorithmically optimized. This ensures that the communications make a strong first impression, no matter who (or what) is reading.

---

## 🎨 Interactive Dashboard

An interactive dashboard visualizing these findings is available [here](https://your-dashboard-link.com) (link to be added once live).
