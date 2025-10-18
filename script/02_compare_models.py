import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
import textstat
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load data
with open('data/email.json', 'r') as f:
    data = json.load(f)
print("....load")
vader = SentimentIntensityAnalyzer()
afinn = Afinn()

# Load NRC Emotion Lexicon
print("Loading NRC Emotion Lexicon...")
url = "https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
nrc = pd.read_csv(url, sep='\t', names=['word', 'emotion', 'score'])
nrc = nrc[nrc['score'] == 1][['word', 'emotion']]

nrc_dict = {}
for word, emotion in nrc.values:
    if word not in nrc_dict:
        nrc_dict[word] = []
    nrc_dict[word].append(emotion)
print(f"✅ Loaded {len(nrc_dict)} words with emotion labels")

# Keywords
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
            'afinn_score': 0,
            'afinn_positive_count': 0,
            'afinn_negative_count': 0,
            'empathy_words': 0,
            'apology_words': 0,
            'personal_pronouns': 0,
            'pronoun_density': 0,
            'empathy_density': 0,
            'mentions_future': False,
            'contains_feedback': False,
            'flesch_reading': None,
            # NRC Emotions
            'emotion_joy': 0,
            'emotion_trust': 0,
            'emotion_anticipation': 0,
            'emotion_sadness': 0,
            'emotion_fear': 0,
            'emotion_anger': 0,
            'emotion_disgust': 0,
            'emotion_surprise': 0,
            'emotion_positive': 0,
            'emotion_negative': 0
        }

    text_lower = text.lower()
    words = re.findall(r'\b[a-z]+\b', text_lower)
    word_count = len(words)

    # Sentiment scores
    vader_score = vader.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity

    # AFINN
    afinn_total = sum(afinn.score(word) for word in words)
    afinn_pos_count = sum(1 for word in words if afinn.score(word) > 0)
    afinn_neg_count = sum(1 for word in words if afinn.score(word) < 0)

    # NRC Emotions
    emotions = {
        'joy': 0, 'trust': 0, 'anticipation': 0, 'sadness': 0,
        'fear': 0, 'anger': 0, 'disgust': 0, 'surprise': 0,
        'positive': 0, 'negative': 0
    }
    
    for word in words:
        if word in nrc_dict:
            for emotion in nrc_dict[word]:
                if emotion in emotions:
                    emotions[emotion] += 1

    # Manual keywords
    empathy_count = sum(1 for word in words if word in empathy_keywords)
    apology_count = sum(1 for word in words if word in apology_keywords)
    pronoun_count = sum(1 for word in words if word in personal_pronouns)

    # Densities
    pronoun_density = pronoun_count / word_count if word_count else 0
    empathy_density = empathy_count / word_count if word_count else 0

    # Boolean features
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
        'afinn_score': afinn_total,
        'afinn_positive_count': afinn_pos_count,
        'afinn_negative_count': afinn_neg_count,
        'empathy_words': empathy_count,
        'apology_words': apology_count,
        'personal_pronouns': pronoun_count,
        'pronoun_density': pronoun_density,
        'empathy_density': empathy_density,
        'mentions_future': mentions_future,
        'contains_feedback': contains_feedback,
        'flesch_reading': flesch_score,
        # NRC Emotions
        'emotion_joy': emotions['joy'],
        'emotion_trust': emotions['trust'],
        'emotion_anticipation': emotions['anticipation'],
        'emotion_sadness': emotions['sadness'],
        'emotion_fear': emotions['fear'],
        'emotion_anger': emotions['anger'],
        'emotion_disgust': emotions['disgust'],
        'emotion_surprise': emotions['surprise'],
        'emotion_positive': emotions['positive'],
        'emotion_negative': emotions['negative']
    }

# Process emails
print("Processing emails...")
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

df = pd.DataFrame(all_emails)
print(f"✅ Processed {len(df)} emails")

# Load transformer models
print("\nLoading transformer models...")

# RoBERTa - 3-class model
MODEL_NAME_ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_NAME_ROBERTA)
model_roberta = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_ROBERTA)

def roberta_sentiment_score(text, tokenizer, model):
    """Calculate normalized sentiment score for 3-class RoBERTa model"""
    if not text or text.strip() == "":
        return None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    
    # For cardiffnlp model: [negative, neutral, positive]
    return probs[2] - probs[0]  # positive - negative

print("Calculating RoBERTa scores...")
df['hf_roberta_score'] = df['email_text'].apply(lambda x: roberta_sentiment_score(x, tokenizer_roberta, model_roberta))

# SST-2 - 2-class model
print("\nLoading SST-2 model...")
MODEL_NAME_SST2 = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sst2 = AutoTokenizer.from_pretrained(MODEL_NAME_SST2)
model_sst2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_SST2)

def sst2_sentiment_score(text, tokenizer, model):
    """Calculate normalized sentiment score for 2-class SST-2 model"""
    if not text or text.strip() == "":
        return None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    
    # For SST-2 model: [negative, positive]
    return probs[1] - probs[0]  # positive - negative

print("Calculating SST-2 scores...")
df['hf_sst2_score'] = df['email_text'].apply(lambda x: sst2_sentiment_score(x, tokenizer_sst2, model_sst2))

# Save extended analysis
print("\nSaving results...")
df.to_csv('data/rejection_analysis_extended.csv', index=False)
print("✅ Saved to data/rejection_analysis_extended.csv")

# Summary statistics
summary = df.groupby('status')[['vader_compound', 'textblob_polarity', 'afinn_score',
                                'hf_roberta_score', 'hf_sst2_score']].agg(['mean','std'])
summary.to_csv('data/rejection_summary.csv')
print("✅ Saved summary to data/rejection_summary.csv")

# Correlation matrix
print("\n📊 CORRELATION MATRIX:")
corr = df[['vader_compound', 'textblob_polarity', 'afinn_score',
           'hf_roberta_score', 'hf_sst2_score']].corr()
corr.to_csv('data/rejection_correlation_compare.csv')
print(corr)

# Emotion correlations
print("\n📊 EMOTION CORRELATIONS WITH VADER:")
emotion_cols = ['emotion_joy', 'emotion_trust', 'emotion_anticipation', 
                'emotion_sadness', 'emotion_fear', 'emotion_anger']
for emotion in emotion_cols:
    corr_val = df[df['status'] != 'ghosted'][emotion].corr(df[df['status'] != 'ghosted']['vader_compound'])
    print(f"  {emotion.replace('emotion_', '').capitalize():15} {corr_val:+.3f}")

# Show model disagreements
print("\n📊 MODEL DISAGREEMENTS:")
df_plot = df[df['status'] != 'ghosted'].copy()
df_plot['vader_roberta_gap'] = abs(df_plot['vader_compound'] - df_plot['hf_roberta_score'])

print("\nTop 5 biggest disagreements (VADER vs RoBERTa):")
print(df_plot.nlargest(5, 'vader_roberta_gap')[['company_id', 'vader_compound', 'hf_roberta_score', 'vader_roberta_gap']])

print("\n✅ Extended analysis complete!")