
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

preprocessor = ColumnTransformer(
    transformers=[
        ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

joblib.dump(model, "model.joblib")

print("Model trained successfully!")
print(f"Model accuracy: {accuracy:.2%}")
print("Saved as model.joblib")
