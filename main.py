import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# טעינה וניקוי
df = pd.read_csv('parkinsons.csv').dropna()

independents = ['MDVP:Shimmer(dB)', 'PPE']
dependent = 'status'
X = df[independents]
y = df[dependent]

# 1. קודם כל מחלקים ל-Train ו-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. מבצעים Scaling בצורה נכונה (רק אם את רוצה לתרגל סקיילר, כאמור בעץ זה לא חובה)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) # לומד מהאימון ומנרמל
X_test_scaled = scaler.transform(X_test)      # מנרמל לפי מה שלמד מהאימון

# 3. אימון המודל
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. חיזוי ובדיקה
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
