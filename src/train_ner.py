'''
Author      : Jeremy S. (@breaktoprotect)
Date created: 8 Apr 2023
Description :
- Read and prepare data for NER training
- Perform NER Training
- Save model into a file
'''
import spacy
import random
from spacy.training.example import Example

def main(): 
    # Initialize 'blank' with English as language model
    nlp = spacy.blank("en")

    # Define the NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        print("[*] NLP Pipeline setup:")
        print(nlp.pipe_names)
        print("")

def load_training_data(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(file)
        

def auto_annotation(train_data):
    pass