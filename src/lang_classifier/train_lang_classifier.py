'''
Author      : Jeremy S. (@breaktoprotect)
Date created: 9 Apr 2023
Description :
- Read and prepare data for programming langauge classifier training
- Perform classification
- Save model into a file
'''
import csv
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.lang.en import English

#? Configuration 
#TODO - put this in a configuration file
TRAIN_DATA_FILENAME = "./training_data/sample_train_data.csv" 
ITERATION_COUNT = 10

def main() -> None:
    # Load a blank model based on English
    nlp = spacy.blank("en")

    # Load the data
    csv_train_data, label_set = load_csv_train_data(TRAIN_DATA_FILENAME)

    # Create a classifier component and add it to the pipelin
    textcat = nlp.add_pipe("textcat", last=True)
    
    # Register each label to the classifier
    for label in label_set:
        textcat.add_label(label)

    # Prepare Spacy train data
    spacy_train_data = prepare_train_data(csv_train_data, label_set)

    #* Perform Text Classification Training of a model to detect programming language
    optimizer = nlp.begin_training()
    for i in range(ITERATION_COUNT):
        random.shuffle(spacy_train_data)
        losses = {}

        batches = minibatch(spacy_train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = []
            for text, annotation in batch:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, annotation))
            nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)

        print(f"Epoch {i+1} Loss: {losses['textcat']}")

    #? Visually test the prediction, see how the model performs
    test_predict(nlp)

    return

def load_csv_train_data(filename:str) -> tuple:
    train_data = []
    label_values = set()
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["cve_text"]
            label = row["prog_lang"]
            label_values.add(label)
            train_data.append((text, label))

    return train_data, label_values

def prepare_train_data(csv_train_data:list, label_values:set) -> list:
    spacy_train_data = []
    for text, label in csv_train_data:
        cats = {l: label == l for l in label_values}
        spacy_train_data.append((text, {"cats": cats}))

    return spacy_train_data

def convert_to_examples():
    nlp = English()
    examples = []
    for text, labels in data:
        cats = {}
        for label in labels:
            cats[label] = 1
        example = Example.from_dict(nlp.make_doc(text), {"cats": cats})
        examples.append(example)
    return examples

def test_predict(nlp_model) -> None:
    test_data = [
        'The PHP library is subjected to xss. It\'s very very bad. Blah blah blah',
        'If one were to use the Java library, it is going to cause a hurt real bad',
    ]
    for text in test_data:
        doc = nlp_model(text)
        max_prob = max(doc.cats, key=doc.cats.get)
        print(f"Text: {text}\nPredicted Label: {max_prob}\n")

if __name__ == "__main__":
    main()