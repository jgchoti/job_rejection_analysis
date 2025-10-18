import pandas as pd
from transformers_interpret import SequenceClassificationExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json


model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cls_explainer = SequenceClassificationExplainer(model, tokenizer)


df = pd.read_csv('data/rejection_analysis_extended.csv')

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'mine',
    'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'a', 'an', 'the',
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'because', 'although', 'though', 'while', 'if', 'unless',
    'of', 'at', 'by', 'with', 'from', 'to', 'in', 'on',
    'about', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'up', 'down', 'between',
    'under', 'over', 'against', 'within', 'without',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must',
    'get', 'got', 'keep', 'kept', 'make', 'made',
    'time', 'role', 'position', 'application', 'job',
    'email', 'name', 'team', 'company', 'page',
    'this', 'that', 'these', 'those',
    'some', 'any', 'much', 'many', 'more', 'most',
    'all', 'both', 'each', 'every', 'either', 'neither',
    'other', 'another', 'such',
    'very', 'really', 'just', 'quite', 'too', 'so',
    'then', 'there', 'here', 'now', 'again',
    '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']',
    "'", '"', '/', '\\', '|',
    "'re", "'ve", "'ll", "'d", "'m", "'t", "n't",
    'ing', 'ed', 'es', 's',
    'name', 'company',
}

MEANINGFUL_LIST = {
    'thank', 'thanks', 'appreciate', 'appreciated', 'grateful',
    'impressed', 'value', 'valued', 'happy', 'pleased',
    'encourage', 'encouraged', 'hope', 'wish', 'best',
    'good', 'great', 'excellent', 'strong', 'competitive',
    'welcome', 'interested', 'opportunity', 'opportunities',
    'sorry', 'apologies', 'apologize', 'unfortunately', 'regret',
    'regrettably', 'disappointing', 'disappointed', 'sad',
    'decided', 'decision', 'not', 'no', 'yes',
}

def is_meaningful_word(word):
    word_lower = word.lower()
    if word_lower in STOPWORDS:
        return False
    if len(word) < 3:
        return False
    if word.startswith(("'", "-", "_")):
        return False
    if word.isdigit():
        return False
    if any(c in word for c in ['[', ']', '(', ')', '{', '}', '<', '>']):
        return False
    return True

def filter_word_attributions(word_attributions):
    filtered = []
    seen_words = set()  
    
    for word, score in word_attributions:
        if word.lower() in seen_words:
            continue
        if is_meaningful_word(word) or word.lower() in MEANINGFUL_LIST:
            filtered.append((word, score))
            seen_words.add(word.lower())
    
    return filtered

print("="*80)
print("🔍 SHAP ANALYSIS WITH SAVE FUNCTIONALITY")
print("="*80)

# Storage for results
all_results = {}
csv_data = []

companies = ['Company_B', 'Company_A', 'Company_F', 'Company_D']

for company in companies:
    print(f"\n{'='*80}")
    print(f"📧 {company}")
    print(f"{'='*80}")
    
    email_data = df[df['company_id'] == company].iloc[0]
    text = email_data['email_text']
    
    # Get word attributions
    word_attributions = cls_explainer(text, class_name="positive")
    meaningful_attrs = filter_word_attributions(word_attributions)
    
    print(f"\nVADER: {email_data['vader_compound']:.3f}")
    print(f"RoBERTa: {email_data['hf_roberta_score']:.3f}")
    

    print(f"\n🟢 Top MEANINGFUL words pushing POSITIVE:")
    print(f"{'Word':<25} {'Attribution':>12}")
    print("-" * 40)
    
    sorted_attrs = sorted(meaningful_attrs, key=lambda x: x[1], reverse=True)
    
    positive_count = 0
    for word, score in sorted_attrs:
        if score > 0 and positive_count < 10:
            bar = "█" * int(score * 50)
            print(f"{word:<25} {score:>12.4f} {bar}")
            positive_count += 1
    
    print(f"\n🔴 Top MEANINGFUL words pushing NEGATIVE:")
    print(f"{'Word':<25} {'Attribution':>12}")
    print("-" * 40)
    
    negative_count = 0
    for word, score in sorted(sorted_attrs, key=lambda x: x[1]):
        if score < 0 and negative_count < 10:
            bar = "█" * int(abs(score) * 50)
            print(f"{word:<25} {score:>12.4f} {bar}")
            negative_count += 1
    
    # Calculate sums
    positive_sum = sum(s for w, s in meaningful_attrs if s > 0)
    negative_sum = sum(s for w, s in meaningful_attrs if s < 0)
    net_meaningful = positive_sum + negative_sum
    
    print(f"\n📊 Meaningful Word Impact:")
    print(f"   Positive sum: {positive_sum:+.3f}")
    print(f"   Negative sum: {negative_sum:+.3f}")
    print(f"   Net impact:   {net_meaningful:+.3f}")
    
    # Store results
    all_results[company] = {
        'vader': float(email_data['vader_compound']),
        'roberta': float(email_data['hf_roberta_score']),
        'positive_sum': float(positive_sum),
        'negative_sum': float(negative_sum),
        'net_impact': float(net_meaningful),
        'words': [(word, float(score)) for word, score in sorted_attrs]
    }
    
    # Add to CSV data
    for word, score in sorted_attrs:
        csv_data.append({
            'company': company,
            'word': word,
            'attribution': score,
            'direction': 'positive' if score > 0 else 'negative',
            'vader_score': email_data['vader_compound'],
            'roberta_score': email_data['hf_roberta_score']
        })

# Save to JSON (complete structure)
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

with open('data/shap_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("✅ Saved to data/shap_results.json")

# Save to CSV (flat structure for easy analysis)
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('data/shap_word_attributions.csv', index=False)
print("✅ Saved to data/shap_word_attributions.csv")

# Save summary statistics
summary_data = []
for company, results in all_results.items():
    summary_data.append({
        'company': company,
        'vader': results['vader'],
        'roberta': results['roberta'],
        'positive_sum': results['positive_sum'],
        'negative_sum': results['negative_sum'],
        'net_impact': results['net_impact'],
        'gap': abs(results['vader'] - results['roberta'])
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/shap_summary.csv', index=False)
print("✅ Saved to data/shap_summary.csv")

print("\nGenerated files:")
print("  📄 data/shap_results.json          - Complete word attributions")
print("  📄 data/shap_word_attributions.csv - Flat CSV for analysis")
print("  📄 data/shap_summary.csv           - Summary statistics")