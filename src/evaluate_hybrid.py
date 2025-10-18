import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

X_train = np.load("outputs/train_embeddings.npy")
y_train = np.load("outputs/train_labels.npy")
X_test = np.load("outputs/test_embeddings.npy")
y_test = np.load("outputs/test_labels.npy")

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Hybrid Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
