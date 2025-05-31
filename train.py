# train.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# 1. Wczytanie i tasowanie danych
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1, random_state=42)

# 2. Podział cech numerycznych i kategorycznych
num_features = ["Age", "Na_to_K"]
cat_features = ["Sex", "BP", "Cholesterol"]

# 3. Definicja transformerów
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 4. Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Podział na zbiór treningowy i testowy
X = drug_df.drop("Drug", axis=1)
y = drug_df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Trenowanie modelu
clf.fit(X_train, y_train)
joblib.dump(clf, "Model/drug_model.pkl")

# 7. Ewaluacja
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 8. Zapis macierzy pomyłek
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig("Results/confusion_matrix.png")

# 9. Zapis raportu do pliku
with open("Results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))
