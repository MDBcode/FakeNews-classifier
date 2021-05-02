import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("train.csv")
df = df.dropna(axis="index")
test_data = pd.read_csv("test.csv")
test_data_text = test_data["text"]

conversion_dict = {0: 'Real', 1: 'Fake'}
df['label'] = df['label'].replace(conversion_dict)

x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.20, random_state=7, shuffle=True)
tfid_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.75)

vec_train = tfid_vectorizer.fit_transform(x_train.values.astype("U"))
vec_test = tfid_vectorizer.transform(x_test.values.astype("U"))
vec_test_data = tfid_vectorizer.transform(test_data_text.values.astype("U"))

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train, y_train)

y_pred = pac.predict(vec_test)
predictions = pac.predict(vec_test_data)
score = accuracy_score(y_test, y_pred)
print(f"PAC Accuracy: {round(score*100,2)}%")
print(confusion_matrix(y_test, y_pred, labels=["Real", "Fake"]))
print(predictions)
