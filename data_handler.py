import os
import csv
from collections import defaultdict
import pandas as pd


def load_data(data_dir, annotation_file, corpus):

    if corpus == "UKP":
        with open(
            os.path.join(data_dir, annotation_file + ".tsv"),
            "r",
            encoding="utf-8",
        ) as f:
            lines = f.readlines()
            for l in lines:
                line = l.split("\t")
                if "_" in line[5]:
                    print(line[4:-1])
    elif corpus == "IBM":
        with open(
            os.path.join(data_dir, annotation_file + ".csv"),
            "r",
            encoding="utf-8",
        ) as f:
            rows = csv.reader(f, delimiter=",", quotechar='"')
            for row in rows:
                print(row[1], row[2])


# data_dir = "data/UKPSententialAM/"
# annotation_file = "gun_control"
# pico_dict = load_data(data_dir, annotation_file, "IBM")


class DataHandler:
    def __init__(self):
        self.data = -1

    def add_columns(self, column_dict, dataframe=None, updateHandler=False):
        """
        Updates a given DataFrame or self.data with new columns from a dictionary
        :param column_dict: dictionary where key is new column name and value is a list of n elements, where n is number of rows in existing DataFrame
        :param dataframe: DataFrame or None (uses self.data instead)
        :param updateHandler: boolean
        :return: updated DataFrame
        """
        if dataframe is not None:
            df = dataframe
        else:
            df = self.data.copy()

        for key in column_dict.keys():
            column = {key: column_dict[key]}
            df = df.assign(**column)

        if updateHandler:
            self.data = df
        return df

    def get_topics(self, corpus=None):
        if not corpus:
            topics = self.data["topic"]
        else:
            topics = self.data["topic"].loc[self.data["corpus"] == corpus]
        return set(topics)

    def get_example(self, n=1, index=-1):
        max = self.data.shape[0]
        if 0 <= index < max:
            df = self.data.iloc[index]  # returns Series
        else:
            # index = randint(0, max)
            # df = self.data.iloc[index]  #returns Series
            df = self.data.sample(n)  # returns DataFrame
        return df

    def get_corpus(self, corpus=None):
        if corpus:
            df = self.data.loc[self.data["corpus"] == corpus]
        else:
            df = self.data
        return df

    def load_data(self, data_dir, corpus):

        if corpus == "UKP":
            data = defaultdict(list)
            path_prefix = os.path.join(os.getcwd(), data_dir)
            for path2file in os.listdir(path_prefix):
                path2file = os.path.join(path_prefix, path2file)
                with open(path2file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for i, l in enumerate(lines):
                        if i == 0:
                            continue
                        (
                            topic,
                            _,
                            _,
                            _,
                            sentence,
                            annotation,
                            setname,
                        ) = l.split("\t")
                        data["topic"].append(topic)
                        data["sentence"].append(sentence)
                        data["annotation"].append(annotation)
                        data["datasplit"].append(setname.replace("\n", ""))
                        data["corpus"].append(corpus)
            df = pd.DataFrame(data)

        elif corpus == "IBM":
            data = defaultdict(list)
            path_prefix = os.path.join(os.getcwd(), data_dir)
            for path2file in os.listdir(path_prefix):
                fullpath2file = os.path.join(path_prefix, path2file)
                with open(fullpath2file, "r", encoding="utf-8") as f:
                    rows = csv.reader(f, delimiter=",", quotechar='"')
                    for i, row in enumerate(rows):
                        if i == 0:
                            continue
                        data["topic"].append(row[1])
                        data["sentence"].append(row[2])
                        data["annotation"].append(row[4])
                        data["datasplit"].append(path2file[:-4])
                        data["corpus"].append(corpus)
            df = pd.DataFrame(data)

        else:
            print("No corpus reader found for '%s'." % (corpus))

        if type(self.data) is int:
            self.data = df
        else:
            self.data = pd.concat([self.data, df])

    def to_pickle(self, filename, dataframe=None):
        """
        saves DataFrame to pickle file or self.data if no DataFrame is given
        :param filename: name of file
        :param dataframe: optional DataFrame object
        :return: None
        """
        path2file = os.path.join(os.getcwd(), filename)
        if dataframe is not None:
            df = dataframe
        else:
            df = self.data
        df.to_pickle(path2file)

    def set_data(self, dataframe):
        self.data = dataframe

    @staticmethod
    def select_by_corpora(df, corpora):
        return df.loc[df["corpus"].isin(corpora)]

    @staticmethod
    def select_by_datasplit(df, datasplit):
        return df.loc[df["datasplit"] == datasplit]

    @staticmethod
    def select_by_keyword(df, label):
        return df.query('sentence.str.contains("%s")' % label, engine="python")

    @staticmethod
    def select_by_label(df, label):
        return df.query("%s in annotation" % label)

    @staticmethod
    def select_by_topic(df, topic):
        # return df.loc[df["topic"] == topic]
        return df.query('topic == "%s"' % topic)

    @staticmethod
    def to_ibm_format(df, perturbation_list, train_only=True):

        if train_only:
            datasplit = ["train"]
        else:
            datasplit = ["train", "test"]

        for x in datasplit:
            split_df = DataHandler.select_by_datasplit(df, x)

            with open(x + ".csv", "w") as out_file:
                csv_writer = csv.writer(out_file, delimiter=",")
                header = [
                    "topic",
                    "the concept of the topic",
                    "candidate",
                    "candidate masked",
                    "label",
                    "wikipedia article name",
                    "wikipedia url",
                ]
                csv_writer.writerow(header)
                for index, row in split_df.iterrows():
                    line = [
                        row["topic"],
                        row["topic"],
                        row["sentence"],
                        row["sentence"],
                        row["annotation"],
                        "placeholder",
                        "placeholder",
                    ]
                    csv_writer.writerow(line)

                    # write perturbations to file
                    for perturbation in perturbation_list:

                        max_length = 6
                        if len(row[perturbation]) < max_length:
                            max_length = len(row[perturbation])

                        for p in range(0, max_length):
                            line[2] = row[perturbation][p]
                            line[3] = row[perturbation][p]
                            csv_writer.writerow(line)

    @staticmethod
    def to_ukp_format(df, perturbation_list, train_only=True):

        if train_only:
            datasplit = ["train"]
        else:
            datasplit = ["train", "test"]

        topics = df.topic.unique()

        for topic in topics:
            topic_df = DataHandler.select_by_topic(df, topic)
            topic = "_".join(topic.split())

            with open(topic + ".tsv", "w") as out_file:
                tsv_writer = csv.writer(out_file, delimiter="\t")
                header = [
                    "topic",
                    "retrievedUrl",
                    "archivedUrl",
                    "sentenceHash",
                    "sentence",
                    "annotation",
                    "set",
                ]
                tsv_writer.writerow(header)
                for index, row in topic_df.iterrows():
                    line = [
                        row["topic"],
                        "placeholder",
                        "placeholder",
                        "placeholder",
                        row["sentence"],
                        row["annotation"],
                        row["datasplit"],
                    ]
                    tsv_writer.writerow(line)

                    # write perturbations to file
                    if row["datasplit"] in datasplit:
                        for perturbation in perturbation_list:

                            max_length = 6
                            if len(row[perturbation]) < max_length:
                                max_length = len(row[perturbation])

                            for p in range(0, max_length):
                                line[4] = row[perturbation][p]
                                tsv_writer.writerow(line)

    @classmethod
    def from_pickle(cls, picklefile):
        dh = DataHandler()
        dh.data = pd.read_pickle(picklefile)
        return dh

    @classmethod
    def map_column_values(cls, df, column, mapping):
        """
        Maps each value of column according to the mapping.
        :param df: DataFrame
        :param column: string
        :param mapping: dictionary (key will be replaced by value)
        :return: DataFrame
        """
        # df[column] = df[column].map(mapping)
        for key in mapping.keys():
            df.loc[df[column] == key, column] = mapping[key]
        return df


if __name__ == "__main__":

    dh = DataHandler()
    data_dir = "data/IBMevidence/"
    dh.load_data(data_dir, "IBM")

    data_dir = "data/UKPSententialAM/"
    dh.load_data(data_dir, "UKP")

    # print(dh.get_example())
    # print(dh.get_topics("UKP"))
    # print(dh.get_corpus("UKP").shape)

    # df = dh.get_corpus()
    # df = dh.select_by_topic(df, "zoos")
    # df = dh.select_by_keyword(df, "http")
    # df = dh.select_by_label(df, [0])
    # print(df.shape)

    # for index, row in df.iterrows():
    #    print(row['annotation'], row['sentence'])

    # path = os.path.join(os.getcwd(), "data/dataset.pkl")
    # dh.to_pickle(path)

    # print(df['test'][2])

    perturbation_pkl = "./data/dataset_perturbations_all.pkl"
    perturbations = [
        # "ner_examples_all_at_once",
        "ner_examples",
        "no_punctuation",
        "adjectives_synonyms",
        "adjectives_sat_synonyms",
        "noun_hyponym",
        # "topic_alternatives",
        "adverbs_examples",
        "speculative_examples",
        "conjunctions_examples",
    ]
    # dh = DataHandler.from_pickle(perturbation_pkl)
    # df = DataHandler.select_by_corpora(dh.data, ["IBM"])
    # DataHandler.to_ukp_format(df, perturbations, train_only=True)
    # DataHandler.to_ibm_format(df, perturbations, train_only=True)
