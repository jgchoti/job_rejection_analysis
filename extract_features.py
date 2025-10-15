import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
import re

with open('data/email.json', 'r') as f:
    data = json.load(f)

vader = SentimentIntensityAnalyzer()

def extract_features(text):
    if text is None:  
        return {
            'email_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'vader_compound': None,
            'textblob_polarity': None,
            'empathy_words': 0,
            'apology_words': 0,
            'personal_pronouns': 0,
            'mentions_future': False,
            'contains_feedback': False,
            'flesch_reading': None
        }
    
    text_lower = text.lower()
    words = text.split()
    

    vader_score = vader.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity

    empathy_keywords = ['thank', 'appreciate', 'grateful', 'hope', 'wish', 'impressed']
    apology_keywords = ['sorry', 'apologies', 'apologize', 'unfortunately', 'regret', 'regrettably']
    personal_pronouns = ['you', 'your', 'yours', 'you\'re', 'you\'ve']
    future_keywords = ['future', 'again', 'next', 'keep in touch', 'stay connected', 'opportunities']
    feedback_keywords = ['because', 'reason', 'based on', 'not convinced', 'stronger', 'more closely']
    
    return {
        'email_length': len(text),
        'word_count': len(words),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'vader_compound': vader_score,
        'textblob_polarity': textblob_score,
        'empathy_words': sum(1 for word in words if any(kw in word.lower() for kw in empathy_keywords)),
        'apology_words': sum(1 for word in words if any(kw in word.lower() for kw in apology_keywords)),
        'personal_pronouns': sum(1 for word in words if word.lower() in personal_pronouns),
        'mentions_future': any(keyword in text_lower for keyword in future_keywords),
        'contains_feedback': any(keyword in text_lower for keyword in feedback_keywords),
        'flesch_reading': textstat.flesch_reading_ease(text) if text else None
    }


all_emails = []

for email in data['rejection_emails']:
    features = extract_features(email['email_text'])
    email.update(features)
    email['status'] = 'rejection'
    all_emails.append(email)

for email in data['feedback_rejection']:
    features = extract_features(email['email_text'])
    email.update(features)
    email['status'] = 'rejection_with_feedback'
    all_emails.append(email)

for email in data['ghosted_applications']:
    features = extract_features(email['email_text'])
    email.update(features)
    all_emails.append(email)


df = pd.DataFrame(all_emails)
df.to_csv('data/rejection_analysis.csv', index=False)

print(f"✅ Processed {len(df)} entries")
print(f"\n📊 Columns generated:")
print(list(df.columns))
print(f"\n📈 Quick stats:")
print(df[['company_id', 'email_length', 'vader_compound', 'empathy_words', 'mentions_future']].head(15))