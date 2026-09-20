import pandas as pd
import numpy as np
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print(df.head())

print(df["Class"].value_counts(normalize=True))

df["Amount_log"] = np.log1p(df["Amount"])
print(df["Amount_log"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])
print(df[["Amount", "Amount_log", "Amount_scaled"]].head())

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
