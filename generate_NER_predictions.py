from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from data_handler import DataHandler


def detect_entities(df, tagger, model):
    """
    This function iterates each sentence in the df dataframe and generates a new dict list
    with the detections from each sentence.

    Parameters
    ----------
    df : dataframe
        dataframe containing the sentences to detect entities from.
    tagger : flair tagger
        the model used to detect NERs from each sentence
    model : str
        determines what kind of data the tagger will be predicting
    Returns
    -------
    ner_df : list
        a list of dicts containing the detections for each sentence.
        This list respects the same order given in the dataframe df.
    """
    ner_df = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sentence = Sentence(row["sentence"])
        tagger.predict(sentence)
        ner = sentence.to_dict(tag_type=model)
        ner.pop("labels")
        ner.pop("text")
        ner_df.append(ner)
    return ner_df


if __name__ == "__main__":

    dh = DataHandler()
    # data_dir = "data/IBMevidence/"
    # dh.load_data(data_dir, "IBM")

    # data_dir = "data/UKPSententialAM/"
    # dh.load_data(data_dir, "UKP")
    dh = dh.from_pickle("./data/dataset.pkl")
    df = dh.get_corpus()
    split = "test"
    df = dh.select_by_datasplit(df, split)
    # load the NER tagger
    model = "ner"
    tagger = SequenceTagger.load(model)
    # Get predictions
    ner_df = detect_entities(df, tagger, model)
    # Add new column 'ner' with predictions to dataframe
    df = dh.add_columns({model: ner_df}, dataframe=df)
    # Save predictions to a pkl file
    dh.to_pickle("dataset.pkl", dataframe=df)
