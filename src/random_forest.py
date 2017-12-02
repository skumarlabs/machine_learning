from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, n_clusters_per_class=2, random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

clf2 = RandomForestClassifier(n_estimators=10, random_state=11)
clf2.fit(X_train, y_train)
predictions2 = clf2.predict(X_test)
print(classification_report(y_test, predictions2))
