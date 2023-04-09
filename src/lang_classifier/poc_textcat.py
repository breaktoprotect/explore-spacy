import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# Load a pre-trained spaCy model suitable for your domain
nlp = spacy.blank("en")

# Create a TextCategorizer component and add it to the pipeline
textcat = nlp.add_pipe("textcat")
textcat.add_label("Java")
textcat.add_label("OS")
textcat.add_label("Golang")

# Load a training dataset as a list of tuples
train_data = [("A vulnerability in the Java Runtime Environment could allow an unauthenticated, remote attacker to execute arbitrary code on a targeted system.", {"cats": {"Java": 1}}),
              ("A vulnerability in the Linux kernel could allow a local attacker to gain elevated privileges on a targeted system.", {"cats": {"OS": 1}}),
              ("A vulnerability in the Golang net/http package could allow a remote attacker to execute arbitrary code on a targeted system.", {"cats": {"Golang": 1}})]

# Convert the training data to a list of Example objects
train_examples = []
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    train_examples.append(example)

# Train the text classification model
n_iter = 10
optimizer = nlp.begin_training()
for i in range(n_iter):
    random.shuffle(train_examples)
    losses = {}
    batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
    print(f"Epoch {i+1} Loss: {losses['textcat']}")

# Evaluate the model on a held-out dataset
test_data = [("A vulnerability in the PHPMailer library could allow an attacker to execute arbitrary code on a targeted system.", {"cats": {"OS": 1}}),
             ("A vulnerability in the Apache Tomcat server could allow a remote attacker to execute arbitrary code on a targeted system.", {"cats": {"Java": 1}}),
             ("A vulnerability in the Go crypto/x509 package could allow a remote attacker to execute arbitrary code on a targeted system.", {"cats": {"Golang": 1}})]
test_examples = []
for text, annotations in test_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    test_examples.append(example)
scores = nlp.evaluate(test_examples)
print(scores)

# Predict 
text = "A vulnerability in the Python requests library could allow an attacker to execute arbitrary code on a targeted system."

doc = nlp(text)

print(doc.cats)