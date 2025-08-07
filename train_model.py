import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load and clean data
df = pd.read_csv("combined_labelled_data.csv")
df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
df = df.dropna(subset=['Label'])

# Normalize labels
df['label'] = df['Label'].str.strip().str.lower().replace({
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

df['clean_text'] = df['text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# TF-IDF
vectorizer = TfidfVectorizer(max_features=30000, stop_words=None, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['clean_text'])
y = df['encoded_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train model
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

# Save TF-IDF + model as pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('xgb', model)
])

# Save the pipeline and label encoder
joblib.dump(pipeline, "xgb_text_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model and vectorizer saved successfully!")
