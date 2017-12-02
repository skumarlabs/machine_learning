import pandas as pd

df = pd.read_csv('./data/SMSSpamCollection', delimiter='\t', header=None)
print(df.head())

print("Number of spam messages: ", df[df[0] == 'spam'][0].count())
print("Number of ham messages: ", df[df[0] == 'ham'][0].count())

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

X = df[1].values
y = df[0].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, random_state=5)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

cls = LogisticRegression()
cls.fit(X_train, y_train)
preds = cls.predict(X_test)
for i, pred in enumerate(preds[:5]):
    print('Predicted: %s, message: %s' % (pred, X_test_raw[i]))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

confusion_mat = confusion_matrix(y_test, preds)
confusion_mat
plt.matshow(confusion_mat)  # TP when cls correctly predicts message is spam
plt.colorbar()
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

from sklearn.metrics import roc_curve, auc

cls2 = LogisticRegression()
cls2.fit(X_train, y_train)
scores = cross_val_score(cls2, X_train, y_train, cv=5)
print('Accuracy: ', scores)
print('Mean accuracy: ', np.mean(scores))

from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()
lbl_encoder.fit(y_train)
y_train_encoded = lbl_encoder.transform(y_train)
y_test_encoded = lbl_encoder.transform(y_test)
precision = cross_val_score(cls2, X_train, y_train_encoded, cv=5, scoring='precision')
print('precision', np.mean(precision))

recalls = cross_val_score(cls2, X_train, y_train_encoded, cv=5, scoring='recall')
print('recall', np.mean(recalls))

f1 = cross_val_score(cls2, X_train, y_train_encoded, cv=5, scoring='f1')
print('F1 score', np.mean(f1))

pred_probs = cls2.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test_encoded, pred_probs[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristics')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend('lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall out')
plt.show()
