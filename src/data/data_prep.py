import pandas as pd
import os

# load data
csv_file = 'Customer_Service_Training.csv'  
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"please ensure{csv_file} exist）")

df = pd.read_csv(csv_file, encoding='utf-8')  

print("=== load data ===")
print("shape:", df.shape)
print("null check:\n", df.isnull().sum())
print("unique intents:", df['intent'].nunique())
print("intent top 5:\n", df['intent'].value_counts().head())

#clean train.csv
print("\n=== clean data ===")
df_clean = df.drop_duplicates(subset=['utterance'], keep='first')  # base on utterance
df_clean['intent'] = df_clean['intent'].str.strip().str.lower()  
df_train = df_clean[['utterance', 'intent']].rename(columns={'utterance': 'query', 'intent': 'label'})

train_path = 'data/train.csv'
df_train.to_csv(train_path, index=False)

print(f"shape after clean: {df_train.shape}")
print("samples:\n", df_train.head(3))

