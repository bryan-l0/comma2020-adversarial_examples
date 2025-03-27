import pandas as pd
import random
import pickle
from tqdm import tqdm
from data_handler import DataHandler
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
from scipy import spatial
import os

NER_TYPE = "type"
NER_TEXT = "text"
NER_START = "start_pos"
NER_END = "end_pos"

BEFORE = "BEFORE"
AFTER = "AFTER"


NUM_MAX_EXAMPLES = 5
NUM_MAX_ADJECTIVES = 2

# Examples of entities_list and adverbs_list.
example_entities_list = {
    "LOC": ["Buenos Aires", "Cordoba"],
    "PER": ["Serena", "Tobias", "Elena", "Santiago"],
    "ORG": ["Audiotechnica", "Bose", "Sony", "Sennheiser"],
    "MISC": ["example_text"],
}

example_adverbs_list = {
    "first": ["however", "additionally"],
    "second": ["yet", "not"],
}


BLACKLIST = ["many", "such"]


def load_adverbs_list(fn):
    with open(fn, "r") as r:
        lines = r.readlines()
    adverbs_list = {BEFORE: [], AFTER: []}
    key = BEFORE
    for l in lines:
        l = l.strip()
        if l == "":
            continue
        if l == BEFORE:
            key = l
            continue
        elif l == AFTER:
            key = l
            continue
        else:
            adverbs_list[key].append(l)
    return adverbs_list


# Load the entities list extracted from the CONNLL2003 dataset
with open("./data/entities_list.pickle", "rb") as handle:
    entities_list = pickle.load(handle)

# Load the speculative adverbs into a list
with open("./data/speculative.txt", "r") as handle:
    speculative_list = handle.readlines()
speculative_list = [adv.strip() for adv in speculative_list]

# Load the conjunctions into a list
with open("./data/conjs.txt", "r") as handle:
    conj_list = handle.readlines()
conj_list = [adv.strip() for adv in conj_list]

# init embedding
fasttext_emb = WordEmbeddings("en")

# Loads the scalar advebs into a list
adverbs_list = load_adverbs_list("./data/scalarity.txt")


def get_vector(word, embed=fasttext_emb):
    """
    Parameters
    ----------
    word : str
        the token to be vectorized
    embed : instance
        the embedding instance used to vectorize
    Returns
    -------
    token_vector : vector
    """
    sentence = Sentence(word)
    embed.embed(sentence)
    for token in sentence:
        return token.embedding


def vectorize_entities(entities, embed):
    vectors_list = {}
    for key in entities.keys():
        vectors_list[key] = {}
        for word in entities[key]:
            vector = get_vector(word, embed)
            vectors_list[key][word] = vector
    return vectors_list


VECTORS_LIST = vectorize_entities(entities_list, fasttext_emb)


def find_closest_embeddings(vectors, embedding):
    return sorted(
        vectors.keys(),
        key=lambda word: spatial.distance.euclidean(vectors[word], embedding),
    )


def update_entities(change, entities):
    for entity in entities:
        entity[NER_START] += change
        entity[NER_END] += change
    return entities


def generate_ner_perturbations(df, entity_name=""):
    """
    This functions generates adversarial examples by exchanging the entities of each sentence.
    You must provide which type of named entity you want to exchange.

    Parameters
    ----------
    df : dataframe
        must contain the column 'sentence' which will be changed to generate new examples.
    entity_name : str
        name of the entity to be replaced. Could be 'PER', 'LOC', 'ORG', 'MISC'
    Returns
    -------
    adversarial_examples : list[examples_list]
        list with same size as number of sentences, with a list of examples for each sentence.
    """
    adversarial_examples = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sent = row["sentence"]
        entities = row["ner"]["entities"]
        adv_examples = []
        for entity in entities:
            ner_type = entity[NER_TYPE]
            if entity_name and ner_type != entity_name:
                pass
            else:
                # change = random.choice(entities_list[ner_type])
                w_vector = get_vector(entity[NER_TEXT])
                candidates = find_closest_embeddings(
                    VECTORS_LIST[ner_type], w_vector
                )
                if entity[NER_TEXT] in candidates:
                    candidates.remove(entity[NER_TEXT])
                entity["candidates"] = candidates[10 : 10 + NUM_MAX_EXAMPLES]
                # for change in candidates:
        og_entities = [en.copy() for en in entities]
        for idx in range(NUM_MAX_EXAMPLES):
            example = sent
            entities = [en.copy() for en in og_entities]
            for entity in entities:
                change = entity["candidates"][idx]
                example = (
                    example[: entity[NER_START]]
                    + change
                    + example[entity[NER_END] :]
                )
                entities = update_entities(
                    len(change) - len(entity[NER_TEXT]), entities
                )
                adv_examples.append(example)
        adversarial_examples.append(adv_examples)
    return adversarial_examples


def generate_ner_examples(df, entity_name=""):
    """
    This functions generates adversarial examples by exchanging the entities of each sentence.
    You must provide which type of named entity you want to exchange.

    Parameters
    ----------
    df : dataframe
        must contain the column 'sentence' which will be changed to generate new examples.
    entity_name : str
        name of the entity to be replaced. Could be 'PER', 'LOC', 'ORG', 'MISC'
    Returns
    -------
    adversarial_examples : list[examples_list]
        list with same size as number of sentences, with a list of examples for each sentence.
    """
    adversarial_examples = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sent = row["sentence"]
        entities = row["ner"]["entities"]
        adv_examples = []
        for entity in entities:
            ner_type = entity[NER_TYPE]
            if entity_name and ner_type != entity_name:
                pass
            else:
                # change = random.choice(entities_list[ner_type])
                w_vector = get_vector(entity[NER_TEXT])
                candidates = find_closest_embeddings(
                    VECTORS_LIST[ner_type], w_vector
                )
                if entity[NER_TEXT] in candidates:
                    candidates.remove(entity[NER_TEXT])
                candidates = candidates[10 : 10 + NUM_MAX_EXAMPLES]
                for change in candidates:
                    example = (
                        sent[: entity[NER_START]]
                        + change
                        + sent[entity[NER_END] :]
                    )
                    adv_examples.append(example)
        adversarial_examples.append(adv_examples)
    return adversarial_examples


def generate_examples(candidates, sent, entity, where=BEFORE):
    """
    Auxiliary function that given some candidates, a sentence, the entity affected and the position,
    generates a new sentence for each candidate by adding each one to the position indicated.
    """
    adv_examples = []
    for change in candidates:
        if where == BEFORE:
            example = (
                sent[: entity[NER_START]]
                + change
                + " "
                + sent[entity[NER_START] :]
            )
        elif where == AFTER:
            example = (
                sent[: entity[NER_END]]
                + " "
                + change
                + " "
                + sent[entity[NER_END] :]
            )
        adv_examples.append(example)
    return adv_examples


def generate_adv_examples(entities, typeof, sent, speculative=False):
    """
    Generates adversarial examples by adding adverbs before or after the verb.
    The adverbs can be added to an adjective (ADJ) or verb (VERB)
    Parameters
    ----------
    entities : dataframe
        df with all the information of the entity.
    typeof : str
        indicates if the adverb must modify the adjective or the verb
    sent : str
        the complete sentence to be modified
    speculative : bool
        flag to indicate if the adverb to be used is from the speculative list or not.
    Returns
    -------
    adversarial_examples : list[examples_list]
        list with same size as number of sentences, with a list of examples for each sentence.
    """

    adv_examples = []
    # Filter entities by its type
    selected_entities = [en for en in entities if en[NER_TYPE] == typeof]
    if typeof == "ADJ":
        # We eliminate the adjectives in the blacklist
        selected_entities = [
            en
            for en in selected_entities
            if en[NER_TEXT].lower() not in BLACKLIST
        ]
        selected_entities = selected_entities[:NUM_MAX_ADJECTIVES]
    elif typeof == "VERB":
        # We only add adverbs to one verb.
        selected_entities = selected_entities[:1]

    for entity in selected_entities:
        if speculative:
            candidates = random.sample(speculative_list, k=2)
        else:
            candidates = random.sample(adverbs_list[BEFORE], k=2)

        examples = generate_examples(candidates, sent, entity, where=BEFORE)
        adv_examples.extend(examples)

    return adv_examples


def gen_conj_examples(entities, sent):
    """
    Generates adversarial examples by adding k conjunctions to the beggining of the sentence

    Parameters
    ----------
    entities : df
        dataframe containing the entities of the sentence.
    sent : str
        sentence to be modified
    Returns
    -------
    adv_examples : list
        list of new adversarial examples
    """
    adv_examples = []
    # Beggining of the sentence
    entity = entities[0]
    if entity[NER_TYPE] != "ADV":
        changes = random.sample(conj_list, k=3)
        for change in changes:
            example = change + ", " + sent
            adv_examples.append(example)
    return adv_examples


def adverbs_perturbations(df):
    """
    This function works as the main function for adverbs. For each sentence, first generates conjunctions examples, then for each sentence
    with adverbs on it, exchanges each with another one from the same list.
    In case there are no adverbs, proceeds to generate adversarial examples by adding new ones, modifiying the adjective and verb.
    Parameters
    ----------
    df : dataframe
    Returns
    -------
    adversarial_examples : list
        list of not speculative adversarial examples
    speculative_examples : list
        list of speculative adversarial examples
    conjunctions_examples : list
        list of conjunctions examples
    """
    adversarial_examples = []
    speculative_examples = []
    conjunctions_examples = []
    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        sent = row["sentence"]
        entities = row["pos"]["entities"]
        adv_examples = []
        spec_examples = []
        conj_examples = gen_conj_examples(entities, sent)
        adverbs_in_sent = [ent for ent in entities if ent[NER_TYPE] == "ADV"]
        exchange_adverb = False
        # Exchanges the adverbs already in the sentence
        for adv in adverbs_in_sent:
            change = ""
            if adv[NER_TEXT] in adverbs_list[BEFORE]:
                change = random.choice(adverbs_list[BEFORE])
            elif adv[NER_TEXT] in adverbs_list[AFTER]:
                change = random.choice(adverbs_list[AFTER])
            if change:
                exchange_adverb = True
                example = (
                    sent[: adv[NER_START]] + change + sent[adv[NER_END] :]
                )
                adv_examples.append(example)
        if not exchange_adverb:
            # In the case there was no adverbs in the sentence, generate new ones
            adv_examples = []
            adv_examples.extend(generate_adv_examples(entities, "ADJ", sent))
            adv_examples.extend(generate_adv_examples(entities, "VERB", sent))
            spec_examples.extend(
                generate_adv_examples(entities, "VERB", sent, speculative=True)
            )
        adversarial_examples.append(adv_examples)
        speculative_examples.append(spec_examples)
        conjunctions_examples.append(conj_examples)
    return adversarial_examples, speculative_examples, conjunctions_examples


if __name__ == "__main__":
    # set True if run all perturbations sequentially otherwise only $perturbation will be processed
    run_all = True
    perturbation = "adverbs_examples"

    input_file = "./data/dataset.pkl"

    if run_all:  # work with data from existing file if possible
        output_file = "./data/dataset_perturbations_all.pkl"
        if os.path.isfile(output_file):
            input_file = output_file
    else:
        output_file = "./data/dataset_perturbations_testonly.pkl"

    print("Loading:", input_file)
    dh = DataHandler.from_pickle(input_file)
    df = dh.select_by_datasplit(dh.data, "test")

    perturbations = {
        "ner_examples": generate_ner_examples,
        "ner_examples_all_at_once": generate_ner_perturbations,
        "adverbs_examples": adverbs_perturbations,
    }

    if run_all:
        for p in perturbations:
            if p == "adverbs_examples":
                adverbs, spec, conj = perturbations[p](df)
                df = dh.add_columns({perturbation: adverbs}, dataframe=df)
                df = dh.add_columns(
                    {"speculative_examples": spec}, dataframe=df
                )
                df = dh.add_columns(
                    {"conjunctions_examples": conj}, dataframe=df
                )
            else:
                examples = perturbations[p](df)
                df = dh.add_columns({p: examples}, dataframe=df)
    else:
        if perturbation == "adverbs_examples":
            adverbs, spec, conj = perturbations[perturbation](df)
            df = dh.add_columns({perturbation: adverbs}, dataframe=df)
            df = dh.add_columns({"speculative_examples": spec}, dataframe=df)
            df = dh.add_columns({"conjunctions_examples": conj}, dataframe=df)
        else:
            examples = perturbations[perturbation](df)
            df = dh.add_columns({perturbation: examples}, dataframe=df)
    dh.to_pickle(output_file, dataframe=df)
