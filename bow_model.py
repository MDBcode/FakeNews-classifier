import pandas as pd
import spacy
from spacy.util import minibatch
import random


def load_data(csv_file, split=0.8):
    data = pd.read_csv(csv_file)
    data = data.dropna(axis="index")
    train_data = data.sample(frac=1, random_state=7)

    texts = train_data.text.values
    labels = [{"FAKE": bool(y), "REAL": not bool(y)}
              for y in train_data.label.values]
    split = int(len(train_data) * split)

    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    return texts[:split], train_labels, texts[split:], val_labels


train_texts, train_labels, val_texts, val_labels = load_data("train.csv")


test_data = pd.read_csv("test.csv")
test_data = test_data.dropna(axis="index")
test_texts = test_data.text.values

nlp = spacy.blank("en")

textcat = nlp.create_pipe(
    "textcat", config={"exclusive_classes": True, "architecture": "bow"})

nlp.add_pipe(textcat)
textcat.add_label("REAL")
textcat.add_label("FAKE")


def train(model, train_data, optimizer):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    batches = minibatch(train_data, size=8)
    for batch in batches:
        texts, labels = zip(*batch)
        model.update(texts, labels, sgd=optimizer, losses=losses)
    return losses


spacy.util.fix_random_seed(1)
random.seed(1)
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
# print(losses['textcat'])


def predict(nlp, texts):
    docs = [nlp.tokenizer(text) for text in texts]
    textcat = nlp.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    predicted_class = scores.argmax(axis=1)
    return predicted_class


predictions = predict(nlp, test_texts)
"""for p, t in zip(predictions, test_texts):
    print(f"{textcat.labels[p]}: {t} \n")"""


def evaluate(model, texts, labels):
    predicted_class = predict(model, texts)
    true_class = [int(label['cats']['FAKE']) for label in labels]
    correct_predictions = predicted_class == true_class
    accuracy = correct_predictions.mean()
    return accuracy


n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")
