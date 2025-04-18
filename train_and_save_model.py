import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("car_evaluation.csv")

# Rename columns
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# Split features and target
X = df.drop(columns="class")
y = df["class"]

# Encode features
onehot_encoder = OneHotEncoder()
X_encoded = onehot_encoder.fit_transform(X)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and preprocessors
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("onehot_encoder.pkl", "wb") as f:
    pickle.dump(onehot_encoder, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
