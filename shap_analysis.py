import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

df = pd.read_csv('data/rejection_analysis_extended.csv')

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cls_explainer = SequenceClassificationExplainer(model, tokenizer)

companies = ['Company_B', 'Company_A', 'Company_F', 'Company_D']

# === Figure setup ===
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

# === Loop through companies ===
for i, company in enumerate(companies):
    email_data = df[df['company_id'] == company].iloc[0]
    text = email_data['email_text']

    # Compute word attributions
    word_attributions = cls_explainer(text, class_name="positive")
    df_temp = pd.DataFrame(word_attributions, columns=['word', 'attribution'])

    # Keep top ±10 influential words (sorted by absolute value)
    df_top = df_temp.sort_values('attribution', key=abs, ascending=False).head(10)
    colors = df_top['attribution'].apply(lambda x: 'seagreen' if x > 0 else 'indianred')

    # Plot
    ax = axes[i]
    ax.barh(df_top['word'], df_top['attribution'], color=colors)
    ax.set_title(f"{company}", fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.invert_yaxis()  # positive at top
    ax.set_xlabel("Attribution")

plt.suptitle("Word Importance by Company (Transformer SHAP Analysis)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/word_importance_all.png", dpi=300)
plt.show()

print("✅ Saved combined figure: visualizations/word_importance_all.png")
