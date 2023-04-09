import spacy
import random
from spacy.training.example import Example

# Load the language model
nlp = spacy.blank("en")

# Define the NER component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
    print("[*] NLP Pipeline setup:")
    print(nlp.pipe_names)

# Add the labels to the NER component
labels = ["LIBRARY", "FUNCTION"]
for label in labels:
    ner.add_label(label)

# Prepare the training data
train_data = [
    ("In the beta library, the x() function is vulnerable to remote code execution.", {"entities": [(7, 10, "LIBRARY"), (25, 27, "FUNCTION")]}),
    ("The omega library's bear() function is subjected to remote code execution.", {"entities": [(4, 8, "LIBRARY"), (12, 17, "FUNCTION")]}),
    ("The y() in the gamma library is vulnerable to SQL injection attacks.", {"entities": [(5, 7, "FUNCTION"), (22, 26, "LIBRARY")]}),
    ("The z() in the alpha library is secure against buffer overflow attacks.", {"entities": [(5, 7, "FUNCTION"), (22, 26, "LIBRARY")]}),
]

# Train the NER model
optimizer = nlp.begin_training()
for iteration in range(30):
    print(f"Starting iteration #{iteration}...")
    losses = {}
    random.shuffle(train_data)
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)

    print(losses)

# Test the NER model
doc = nlp("In the beta library, the heh() function is vulnerable to remote code execution. Then the blah() is actually not affected.")
for ent in doc.ents:
    print(ent.text, ent.label_)

