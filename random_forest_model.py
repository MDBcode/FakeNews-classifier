from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df = df.dropna(axis="index")

x = df.drop('label', axis=1)
y = df['label']

x = x.drop(['id'], axis=1)
x.reset_index(inplace=True)

headline = []
for row in range(len(x.title)):
    headline.append(' '.join(str(word).lower() for word in x.iloc[row, 0:2]))

cv = TfidfVectorizer()
X = cv.fit_transform(headline)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

randomclassifier = RandomForestClassifier(
    n_estimators=200, criterion='entropy')
randomclassifier.fit(X_train, y_train)

y_pred = randomclassifier.predict(X_test)
confusion_m = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)

print(confusion_m)
print('\n')
print(accuracy_score)
print(y_pred)
