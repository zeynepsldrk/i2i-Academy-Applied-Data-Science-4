import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

print("Veri seti kagglehub kullanilarak indiriliyor...")
dataset_path = kagglehub.dataset_download("rafi003/student-lifestyle-and-academic-performance-dataset")
file_path = os.path.join(dataset_path, "student_lifestyle_performance_dataset.csv")

try:
    df = pd.read_csv(file_path)
    print("Veri seti basariyla yüklendi!")
except FileNotFoundError:
    print("HATA: Veri seti indirilen dizinde bulunamadi.")
    exit()

if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)

df['Performance_Class'] = df['CGPA'].apply(lambda x: 1 if x >= 7.0 else 0)

label_enc = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = label_enc.fit_transform(df[col])

X = df.drop(columns=['CGPA', 'Internal_Marks', 'Performance_Class'])
y = df['Performance_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)

print("-" * 50)
print("Lojistik Regresyon Sonuclari:")
print("Dogruluk Puani: {:.4f}".format(accuracy_score(y_test, y_pred_log)))
print("Karisiklik Matrisi:")
print(confusion_matrix(y_test, y_pred_log))
print("-" * 50)

print("Rastgele Orman Sonuclari:")
print("Dogruluk Puani: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("Karisiklik Matrisi:")
print(confusion_matrix(y_test, y_pred_rf))
print("-" * 50)
