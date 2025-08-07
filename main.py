import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ------------------
# Load and Preprocess
# ------------------
df = pd.read_csv("combined_labelled_data.csv")
df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
df = df.dropna(subset=['Label'])

# Label Cleaning
df['label'] = df['Label'].str.strip().str.lower()
df['label'] = df['label'].replace({
    'drug and alcohol': 'drug_alcohol',
    'drug and Alcohol': 'drug_alcohol',
    'Drug and alcohol': 'drug_alcohol',
    'Drug and Alcohol': 'drug_alcohol',
    'early life': 'early_life',
    'Early life': 'early_life',
    'Early Life': 'early_life',
    'personality': 'personality',
    'Personality': 'personality',
    'trauma and stress': 'trauma_stress',
    'Trauma and Stress': 'trauma_stress',
    'Trauma and Stress ': 'trauma_stress'
})

# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Encode labels to integers
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# ------------------
# TF-IDF Vectorization + Train-Test Split
# ------------------
vectorizer = TfidfVectorizer(max_features=30000, stop_words=None, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['clean_text'])
y = df['encoded_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ------------------
# XGBoost Classifier
# ------------------
model = XGBClassifier(
    eval_metric='mlogloss',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Decode predictions and actuals back to original labels
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# ------------------
# Evaluation
# ------------------
acc = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n🔍 Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


