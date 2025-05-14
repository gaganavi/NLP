import pandas as pd

df = pd.read_csv(r"C:\Users\gagan\Downloads\archive (1)\train.csv")
print(df.columns)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv(r"C:\Users\gagan\Downloads\archive (1)\train.csv")

df.columns = df.columns.str.strip()

df['title'] = df['title'].fillna('')
df['formatted_prompt'] = df['formatted_prompt'].fillna('')
df['notes'] = df['notes'].fillna('')

df['combined'] = df['title'] + " " + df['formatted_prompt'] + " " + df['notes']

df['combined'] = df['combined'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

df['combined'] = df['combined'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

print("Preprocessing complete. Sample output:")
print(df[['title', 'combined']].head())
