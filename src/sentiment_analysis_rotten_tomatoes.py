import numpy as np
import pandas as pd

df = pd.read_csv('./data/moviereviewtrain.tsv', delimiter='\t', header=0)
print(df.count())

print(df['Phrase'].head(10))  # features
print(df['Sentiment'].head(10))  # target classes

print(df['Sentiment'].describe())
print(df['Sentiment'].value_counts())
print(df['Sentiment'].value_counts() / df['Sentiment'].count())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X, y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': (0.25, .5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10)
}

grid_seach = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_seach.fit(X_train, y_train)

print('Best score: %0.3f' % grid_seach.best_score_)
print('Best parameter set: ')
best_parameters = grid_seach.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print('t%s %r' % (param_name, best_parameters[param_name]))

predictions = grid_seach.predict(X_test)
print('Accuracy: %s' % accuracy_score(y_test, predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))
print('Classification Report')
print(classification_report(y_test, predictions))

X_test_final = pd.read_csv('./data/moviereviewtest.tsv', delimiter='\t', header=0)
X_test_pid, X_test_final = X_test_final['PhraseId'], X_test_final['Phrase']
final_pred = grid_seach.predict(X_test_final)
with open('sentiment_result', 'w') as f:
    f.write('PhraseId,Sentiment\n')
    for i, rating in enumerate(final_pred):
        f.write(str(X_test_pid[i]) + ',' + str(rating) + '\n')
