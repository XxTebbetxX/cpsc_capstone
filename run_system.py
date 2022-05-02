import spacy
import os
import random
from Bio import Entrez
import numpy as np
from SplitClassifier import *

# For use, run the following commands:
    # pip install scispacy
    # pip install spacy
    # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
    # pip install biopython

def determine_question(question):
    # valid question?
    # check for a question mark
    # check for a valid beginning
    # check for proper sentence structure
    total_tests = 1
    passed_tests = 0
    if (question.strip()[-1] == '?'):
        passed_tests += 1

    exact_answer = passed_tests/total_tests
    return np.round(exact_answer)

def determine_yesno(question):
    total_tests = 1
    passed_tests = 0
    starters = ["DO", "DID", "IS", "CAN", "WAS", "COULD", "WOULD", "SHOULD",
                "WILL", "ARE", "IF", "DOES"]
    first_word = question.strip().split(" ")[0]
    if (first_word.upper() in starters):
        passed_tests += 1

    exact_answer = passed_tests / total_tests
    return np.round(exact_answer)


def determine_bio(question):
    total_tests = 0
    passed_tests = 0

    exact_answer = passed_tests / total_tests
    return np.round(exact_answer)

def form_query(question):
    # Query formulation
    print("Forming query...")
    #spacy.SPACY_WARNING_IGNORE =
    nlp = spacy.load("en_core_sci_lg")
    keywords = list(nlp(question).ents)
    query = ''
    for item in keywords:
        query = query + str(item) + " "
    return query[:-1]

def retrieve_passage(query):
    print("Retrieving snippet...")
    Entrez.email = 'tatianna.benjamin.20@cnu.edu'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=str(1),
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    id_list = results['IdList']
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    papers = Entrez.read(handle)
    snippet = ""
    keywords = query.split()
    for num in range(1):
      snippet += ("\n{}\n".format(papers["PubmedArticle"][num]["MedlineCitation"]["Article"]["ArticleTitle"]))
      for paper in papers["PubmedArticle"]:
        for section in paper["MedlineCitation"]["Article"]["Abstract"]:
          if section == "AbstractText":
            for element in paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]:
                valid_snippet = True
                for word in keywords:
                    if word not in element:
                        valid_snippet = False
                if valid_snippet:
                    snippet += "{}\n".format(element)






    return snippet

def get_answer(snippet, question):
    print("Determining answer...")
    language_model_name = SplitClassifier.BASEBERT
    max_length = 512
    rate = .00000001
    filepath = os.path.join("..", "weights", "weights_Split_CLS_softmax8")
    classifier = Split_CLS_softmax(language_model_name, max_length=max_length, language_model_trainable=True,
                                 learning_rate=rate)
    classifier.load_weights(filepath)
    exact_answer = classifier.predict(question, snippet)[0][0]
    if (exact_answer > .5):
        answer = 'Yes'
    else:
        answer = 'No'
    print("\n\nQuestion: {}\nContext: {}\nAnswer: {}".format(question, snippet, answer))
    print("Exact value: {}".format(exact_answer))


if __name__ == '__main__':
    # test_eval = "yesno_dataset.tsv"
    # evaluate_question(test_eval)
    valid_input = False
    print("\n\n\n\t\t\tBiomedical Yes/No QA System\n")
    while not valid_input:
        question = input("Please enter a Yes/No biomedical question: ")
        if determine_question(question) == 1:
            print("Satisfied question requirements.")
            if determine_yesno(question) == 1:
                print("Satisfied yesno requirements.")
                query = form_query(question)
                snippet = retrieve_passage(query)
                # If query and pubmed returns valid results, it is a bio question
                # else, it is not
                print("Satisfied biomedical requirements.")
                valid_input = True
                # Need to incorporate Classifier to predict answer
                get_answer(retrieve_passage(query), question)

            else:
                print("Error. Question has answer other than yes or no. \nPlease enter a valid question or press X to exit.")
                break
            break
        elif question.upper() == "X":
            exit()
        else:
            print("Error. Question not detected. Please enter a valid question or press X to exit.")


    # Incorporate Classifier


    # Output: I am __% confident that the answer is:
