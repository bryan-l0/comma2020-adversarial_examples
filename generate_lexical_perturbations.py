from functools import partial
import os
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from data_handler import DataHandler

ADJ = "JJ"
NOUN = "NN"
NUM_MAX_EXAMPLES = 5


def get_synonyms(word, pos="a"):
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    synonyms = set()
    hyponyms = set()
    syns = wn.synsets(word)
    for s in syns:
        if s.pos() == pos:
            for l in s.lemmas():
                if l.name() != word and l.name() != word.lower():
                    synonyms.add(l.name())
            for h in s.hyponyms():
                hyponyms.add(h.name())

    # print("Synonyms:", synonyms)
    # print("Hyponums:", hyponyms)
    synonyms = list(synonyms)
    hyponyms = list(hyponyms)
    return synonyms, hyponyms


def get_synonyms_from_file(path_to_file="data/topic_synonyms.tsv"):

    synonyms = {}

    with open(path_to_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        for l in lines:
            topic, syns = l.split("\t")
            synonyms[topic] = syns.replace("\n", "").split(",")
    return synonyms


def replace_postag(df, pos="a", replace_with="synonym"):
    """
    Iterates each sentence in df and replace all adjectives or nouns
    (depending on pos) with synonyms or hyponyms (depending on replace_with)
    Parameters
    ----------
    df : dataframe
        dataframe with the sentences to be processed.
    pos : str
        determines the kind of pos tag to exchange.
    replace_with : str
        "synonym" or "hyponym"

    Returns
    -------
    adversarial_examples : list
        list of new generated examples.
    """
    adversarial_examples = []
    if pos == "a" or pos == "s":
        TAG = ADJ
    else:
        TAG = NOUN
    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        sent = row["sentence"]
        tokens = word_tokenize(sent)
        tags = pos_tag(tokens)
        # adjectives: list with index in sentence and the word to be exchanged.
        words = [
            (idx, word) for idx, (word, tag) in enumerate(tags) if tag == TAG
        ]
        examples = []
        for idx, word in words:
            synonyms, hyponyms = get_synonyms(word, pos=pos)
            if replace_with == "synonym":
                candidates = synonyms
            elif replace_with == "hyponym":
                candidates = hyponyms
            candidates = candidates[:NUM_MAX_EXAMPLES]
            new_example = tokens.copy()
            for change in candidates:
                new_example[idx] = change
                new_example = " ".join(new_example)
                examples.append(new_example)
        adversarial_examples.append(examples)
    return adversarial_examples


def replace_topics(df):
    """
    Adds a list of synonyms for each topic in the column "topic_alternatives"
    :param df: DataFrame
    :return: list of length n, where n = len(df.index)
    """
    adversarial_examples = []

    topic_dictionary = get_synonyms_from_file()

    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        if row["topic"] in topic_dictionary.keys():
            alternatives = [x for x in topic_dictionary[row["topic"]]]
        else:
            alternatives = []
        adversarial_examples.append(alternatives)
    return adversarial_examples


def delete_punctuation(df):

    punctuation = [".", ",", ":", ";", "?", "!", '"']

    adversarial_examples = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sent = row["sentence"]
        adv_examples = []
        for punct in punctuation:
            sent = sent.replace(punct, "")
        adv_examples.append(sent)
        adversarial_examples.append(adv_examples)

    return adversarial_examples


if __name__ == "__main__":

    # set True if run all perturbations sequentially otherwise only $perturbation will be processed
    run_all = True
    perturbation = "topic_alternatives"

    input_file = "./data/dataset.pkl"

    if run_all:  # work with data from existing file if possible
        output_file = "./data/dataset_perturbations_all.pkl"
        if os.path.isfile(output_file):
            input_file = output_file
    else:
        output_file = "./data/lexical_perturbations.pkl"

    print("Loading:", input_file)
    dh = DataHandler.from_pickle(input_file)
    # df = dh.select_by_datasplit(dh.data, "test")
    df = dh.data

    perturbations = {
        "no_punctuation": delete_punctuation,
        "adjectives_sat_synonyms": partial(replace_postag, pos="s"),
        "adjectives_synonyms": partial(replace_postag, pos="a"),
        "noun_hyponym": partial(replace_postag, pos="n"),
        "topic_alternatives": replace_topics,
    }

    if run_all:
        for p in perturbations:
            examples = perturbations[p](df)
            df = dh.add_columns({p: examples}, dataframe=df)
        dh.to_pickle(output_file, dataframe=df)
    else:
        examples = perturbations[perturbation](df)
        df = dh.add_columns({perturbation: examples}, dataframe=df)
        dh.to_pickle(output_file, dataframe=df)
