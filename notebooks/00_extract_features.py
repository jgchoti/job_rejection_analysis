import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
import textstat
import re

# Load data
with open('data/email.json', 'r') as f:
    data = json.load(f)

vader = SentimentIntensityAnalyzer()
afinn = Afinn()

# Manual keywords (for interpretable features)
empathy_keywords = ['thank', 'appreciate', 'grateful', 'hope', 'wish', 'impressed']
apology_keywords = ['sorry', 'apologies', 'apologize', 'unfortunately', 'regret', 'regrettably']
future_keywords = ['future', 'again', 'next', 'keep in touch', 'stay connected', 'opportunities']
personal_pronouns = ['you', 'your', 'yours', "you're", "you've"]
feedback_keywords = ['because', 'reason', 'based on', 'not convinced', 'stronger', 'more closely']

def extract_features(text):
    if not text or text.strip() == "":
        return {
            'email_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'vader_compound': None,
            'textblob_polarity': None,
            'afinn_score': 0,  # NEW: AFINN total score
            'afinn_positive_count': 0,  # NEW: Count of positive words
            'afinn_negative_count': 0,  # NEW: Count of negative words
            'empathy_words': 0,
            'apology_words': 0,
            'personal_pronouns': 0,
            'pronoun_density': 0,
            'empathy_density': 0,
            'mentions_future': False,
            'contains_feedback': False,
            'flesch_reading': None
        }
    
    text_lower = text.lower()
    # Clean tokenization - remove punctuation
    words = re.findall(r'\b[a-z]+\b', text_lower)
    word_count = len(words)
    
    # Sentiment scores
    vader_score = vader.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity
    
    # AFINN analysis (using actual AFINN scores)
    afinn_total = 0
    afinn_pos_count = 0
    afinn_neg_count = 0
    
    for word in words:
        score = afinn.score(word)
        afinn_total += score
        if score > 0:
            afinn_pos_count += 1
        elif score < 0:
            afinn_neg_count += 1
    
    # Manual keyword counts (for interpretability)
    empathy_count = sum(1 for word in words if word in empathy_keywords)
    apology_count = sum(1 for word in words if word in apology_keywords)
    pronoun_count = sum(1 for word in words if word in personal_pronouns)
    
    # Densities
    pronoun_density = pronoun_count / word_count if word_count else 0
    empathy_density = empathy_count / word_count if word_count else 0
    
    # Structural features
    mentions_future = any(keyword in text_lower for keyword in future_keywords)
    contains_feedback = any(keyword in text_lower for keyword in feedback_keywords)
    
    # Readability
    flesch_score = textstat.flesch_reading_ease(text) if text else None
    
    return {
        'email_length': len(text),
        'word_count': word_count,
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'vader_compound': vader_score,
        'textblob_polarity': textblob_score,
        'afinn_score': afinn_total,  # Total AFINN sentiment
        'afinn_positive_count': afinn_pos_count,  # Number of positive words
        'afinn_negative_count': afinn_neg_count,  # Number of negative words
        'empathy_words': empathy_count,
        'apology_words': apology_count,
        'personal_pronouns': pronoun_count,
        'pronoun_density': pronoun_density,
        'empathy_density': empathy_density,
        'mentions_future': mentions_future,
        'contains_feedback': contains_feedback,
        'flesch_reading': flesch_score
    }

# Process emails
all_emails = []

for email in data.get('rejection_emails', []):
    features = extract_features(email.get('email_text'))
    email.update(features)
    email['status'] = 'rejection'
    all_emails.append(email)

for email in data.get('feedback_rejection', []):
    features = extract_features(email.get('email_text'))
    email.update(features)
    email['status'] = 'rejection_with_feedback'
    all_emails.append(email)

for email in data.get('ghosted_applications', []):
    features = extract_features(email.get('email_text'))
    email.update(features)
    email['status'] = 'ghosted'
    all_emails.append(email)

# Save to CSV
df = pd.DataFrame(all_emails)
df.to_csv('data/rejection_analysis.csv', index=False)

print(f"✅ Processed {len(df)} entries")
print(f"\n📊 Columns generated: {list(df.columns)}")
print(f"\n📈 Quick stats:")
print(df[['company_id', 'vader_compound', 'afinn_score', 'empathy_words', 'apology_words']].head(10))

print(f"Correlation VADER vs AFINN: {df['vader_compound'].corr(df['afinn_score']):.3f}")
print(f"Correlation Empathy words vs AFINN positive count: {df['empathy_words'].corr(df['afinn_positive_count']):.3f}")