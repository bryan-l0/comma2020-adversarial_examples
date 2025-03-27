import argparse
import json
import os
import numpy as np
import pandas as pd
import pprint
from collections import defaultdict
from ukp_bert_repo.inference import BertEvaluator
from data_handler import DataHandler


def evaluate_model(model_path, datahandler):
    topics = [
        "abortion",
        "cloning",
        "death penalty",
        "gun control",
        "marijuana legalization",
        "minimum wage",
        "nuclear energy",
        "school uniforms",
    ]
    label_list = ["NoArgument", "Argument_against", "Argument_for"]
    max_seq_length = 64
    eval_batch_size = 8

    be = BertEvaluator()
    be.load_model(model_path, label_list)

    results = {}
    prec_results = {}
    rec_results = {}
    for topic in topics:
        filename = "filename"
        print("\n\n===================== " + topic + " ====================")
        df = DataHandler.select_by_datasplit(dh.data, "test")
        df = DataHandler.select_by_topic(df, topic)
        print("DATA SIZE:", len(df.index))
        input_examples = be.get_examples_from_dataframe(df)
        dl = be.get_dataloader(input_examples, max_seq_length=64, batch_size=8)
        preds = be.get_predictions(dl)
        df = datahandler.add_columns({"prediction": preds}, dataframe=df)
        f1, prec, rec, acc = be.ukp_evaluate(df)

        if topic not in results:
            results[topic] = {}
            prec_results[topic] = {}
            rec_results[topic] = {}

        if filename not in results[topic]:
            results[topic][filename] = []
            prec_results[topic][filename] = []
            rec_results[topic][filename] = []

        results[topic][filename].append(f1)
        prec_results[topic][filename].append(prec)
        rec_results[topic][filename].append(rec)

    print("\n\n===================== Overall results ====================")
    model_f1 = {}
    model_prec = {}
    model_rec = {}

    for topic in sorted(results.keys()):
        print(topic)
        for filename in sorted(results[topic].keys()):
            topic_f1_mean = np.mean(results[topic][filename])
            print(
                "%s (%d): %.4f"
                % (filename, len(results[topic][filename]), topic_f1_mean)
            )

            if filename not in model_f1:
                model_f1[filename] = []
                model_prec[filename] = []
                model_rec[filename] = []

            model_f1[filename].append(results[topic][filename])
            for prec in prec_results[topic][filename]:
                model_prec[filename].append(prec)

            for rec in rec_results[topic][filename]:
                model_rec[filename].append(rec)
        print("")

    print("\n\n==========================================")
    print("Average F1 score over all topics")
    for filename in model_f1:
        print(
            "%s (%d) %.2f%%"
            % (
                filename,
                len(model_f1[filename]),
                np.mean(model_f1[filename]) * 100,
            )
        )

    print("\n\n==========================================")
    print("P_arg score over all topics")
    for filename in model_prec:
        prec_pos = [
            prec_result["Argument_for"] for prec_result in model_prec[filename]
        ]
        print(
            "P_arg+ %s (%d): %.4f"
            % (filename, len(prec_pos), np.mean(prec_pos))
        )

        prec_neg = [
            prec_result["Argument_against"]
            for prec_result in model_prec[filename]
        ]
        print(
            "P_arg- %s (%d): %.4f"
            % (filename, len(prec_neg), np.mean(prec_neg))
        )

    print("\n\n==========================================")
    print("R_arg score over all topics")
    for filename in model_rec:
        rec_pos = [
            rec_result["Argument_for"] for rec_result in model_rec[filename]
        ]
        print(
            "R_arg+ %s (%d): %.4f" % (filename, len(rec_pos), np.mean(rec_pos))
        )

        rec_neg = [
            rec_result["Argument_against"]
            for rec_result in model_rec[filename]
        ]
        print(
            "R_arg- %s (%d): %.4f" % (filename, len(rec_neg), np.mean(rec_neg))
        )


def evaluate_perturbation(
    evaluator,
    dataframe,
    perturbation,
    outputfile="perturbation_performance.json",
    predictions_file="perturbation_predictions.pkl",
    topic=None,
):
    """
    Creates for each element in the perturbation cell one adversarial example and evaluates how many percent changed the prediction label.
    :param evaluator: BertEvaluator with an already loaded model
    :param dataframe: DataFrame
    :param perturbation: str perturbation type (must be a column name of dataframe)
    :param outputfile: str name of output file
    :param predictions_file: str name of pickle file to store predictions
    :return: dictionary of performance values
    """

    dh = DataHandler()
    if topic in dataframe.topic.unique():
        df = DataHandler.select_by_topic(dataframe, topic)
    else:
        df = dataframe

    used_corpora = list(df.corpus.unique())
    used_topics = list(df.topic.unique())
    labels = list(df.annotation.unique())

    num_examples = len(df.index)

    df_dict = {
        "topic": [],
        "sentence": [],
        "annotation": [],
        "original_pred": [],
        "corpus": [],
        "perturbated": [],
    }
    for index, row in df.iterrows():

        perturbations = row[perturbation]
        for p in perturbations:
            df_dict["annotation"].append(row["annotation"])
            df_dict["original_pred"].append(row["prediction"])
            df_dict["topic"].append(row["topic"])
            df_dict["perturbated"].append(p)
            df_dict["sentence"].append(row["sentence"])
            df_dict["corpus"].append(row["corpus"])
    df = pd.DataFrame.from_dict(df_dict)
    num_perturbations = len(df.index)

    input_examples = evaluator.get_examples_from_perturbated_df(
        df, perturbation
    )
    dl = evaluator.get_dataloader(input_examples)
    preds = evaluator.get_predictions(dl)
    df = dh.add_columns({"prediction": preds}, dataframe=df)

    # successful = 0
    # unsuccessful = 0
    attacks = {label: defaultdict(int) for label in labels}

    for index, row in df.iterrows():
        if row["original_pred"] == row["prediction"]:
            attacks[row["annotation"]]["unsuccessful"] += 1
        else:
            attacks[row["annotation"]]["successful"] += 1

    avg_num_perturbation = np.round(num_perturbations / num_examples, 2)

    successful_total = int(
        np.sum([attacks[label]["successful"] for label in attacks.keys()])
    )
    unsuccessful_total = int(
        np.sum([attacks[label]["unsuccessful"] for label in attacks.keys()])
    )

    for label in labels:
        attacks[label]["efficiency"] = np.round(
            attacks[label]["successful"]
            / (attacks[label]["successful"] + attacks[label]["unsuccessful"]),
            4,
        )

    perturbation_efficiency = np.round(
        successful_total / (successful_total + unsuccessful_total), 4
    )

    if os.path.isfile(outputfile):
        with open(outputfile, "r", encoding="utf-8") as fin:
            performance_data = json.load(fin)
    else:
        performance_data = {}
    if os.path.isfile(predictions_file):
        pred_dh = DataHandler.from_pickle(predictions_file)
        df = dh.add_columns(
            {"perturbation": [perturbation for _ in range(0, len(df.index))]},
            df,
        )
        df = pred_dh.data.append(df).drop_duplicates()
        pred_dh.to_pickle(predictions_file, df)

    else:
        df = dh.add_columns(
            {"perturbation": [perturbation for _ in range(0, len(df.index))]},
            df,
        )
        dh.to_pickle(predictions_file, df)

    performance_data[perturbation] = {
        "corpus_types": used_corpora,
        "topics": used_topics,
        "size of test set": num_examples,
        "num of generations": num_perturbations,
        "average generation": avg_num_perturbation,
        "successful attacks": successful_total,
        "unsuccessful attacks": unsuccessful_total,
        "Perturbation efficiency": perturbation_efficiency,
        "Specific efficiencies": attacks,
    }

    with open(outputfile, "w", encoding="utf-8") as fout:
        json.dump(performance_data, fout)

    return performance_data


def pretty_print_json(json_file="perturbation_performance.json"):

    pp = pprint.PrettyPrinter()
    with open(json_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
        pp.pprint(data)


def run_evaluation(
    model_path, perturbation_file, corpora, testrun=(False, None)
):

    # load generated perturbations
    dh = DataHandler.from_pickle(perturbation_file)
    dh.set_data(dh.select_by_corpora(dh.data, corpora))

    # load model
    label_list = ["NoArgument", "Argument_against", "Argument_for"]
    label_map = {
        "0": "Argument_for",  # TODO check label mapping in original repo
        "1": "NoArgument",
    }
    df = DataHandler.map_column_values(dh.data, "annotation", label_map)
    df = DataHandler.select_by_datasplit(df, "test")
    be = BertEvaluator()
    be.load_model(model_path, label_list)

    print("Column names:", dh.data.keys())
    print()

    if testrun[0]:
        df = dh.get_example(10)
        output_json = "perturbation_performance.json"
        output_pkl = "perturbation_predictions.pkl"
        results = evaluate_perturbation(
            be,
            df,
            testrun[1],
            outputfile=output_json,
            predictions_file=output_pkl,
        )
        pretty_print_json(output_json)
    else:
        output_json = "output/perturbation_performance.json"
        output_pkl = "output/perturbation_predictions.pkl"
        for i, p in enumerate(perturbations):
            print("Perturbation %s of %s: %s" % (i + 1, len(perturbations), p))
            evaluate_perturbation(
                be, df, p, outputfile=output_json, predictions_file=output_pkl
            )
        pretty_print_json(output_json)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        required=True,
        help="List of corpora.",
    )
    parser.add_argument(
        "--topic", type=str, required=True, help="Name of topic."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to PyTorch Bert model.",
    )
    parser.add_argument(
        "--perturbations", nargs="+", help="List of perturbation types."
    )
    parser.add_argument(
        "--perturbations_file",
        default="./data/dataset_perturbations_all.pkl",
        help="Path to pkl file where perturbations are stored.",
    )
    parser.add_argument("--test_mode", default=False, type=bool)
    args = parser.parse_args()

    if args.perturbations:
        perturbations = args.perturbations
    else:
        perturbations = [
            "ner_examples_all_at_once",
            "ner_examples",
            "no_punctuation",
            "adjectives_synonyms",
            "adjectives_sat_synonyms",
            "noun_hyponym",
            "topic_alternatives",
            "adverbs_examples",
            "speculative_examples",
            "conjunctions_examples",
        ]

    # set True for fast run mode for developing and select perturbation to evaluate
    runAsTest = args.test_mode

    # model_path = 'ukp_bert_repo/bert_model/argument_classification_ukp_all_data/'
    # perturbation_pkl = "./data/dataset_perturbations_testonly.pkl"
    # corpora = ["UKP"]

    # run_evaluation(model_path, perturbation_pkl, corpora, perturbations, runAsTest)

    # load generated perturbations
    dh = DataHandler.from_pickle(args.perturbations_file)
    dh.set_data(dh.select_by_corpora(dh.data, args.corpora))

    # load model
    if "UKP" in args.corpora:
        label_list = ["NoArgument", "Argument_against", "Argument_for"]
        label_map = {
            "0": "Argument_for",  # TODO check label mapping in original repo
            "1": "NoArgument",
        }
        df = DataHandler.map_column_values(dh.data, "annotation", label_map)
    else:
        label_list = ["0", "1"]
        df = dh.data

    df = DataHandler.select_by_datasplit(df, "test")
    be = BertEvaluator()
    be.load_model(args.model_dir, label_list)

    # select only examples which are classified correctly
    ie = be.get_examples_from_dataframe(df)
    dl_tmp = be.get_dataloader(ie)
    preds = be.get_predictions(dl_tmp)
    df = dh.add_columns({"prediction": preds}, dataframe=df)
    # df = df[df.annotation==df.prediction]

    print("Column names:", dh.data.keys())
    print()

    if runAsTest:

        # evaluate_model(args.model_dir, dh)

        df = dh.get_example(25)
        output_json = (
            "perturbation_performance_"
            + args.corpora[0]
            + "_"
            + args.topic
            + ".json"
        )
        output_pkl = (
            "perturbation_predictions_"
            + args.corpora[0]
            + "_"
            + args.topic
            + ".pkl"
        )
        results = evaluate_perturbation(
            be,
            df,
            "ner_examples",
            outputfile=output_json,
            predictions_file=output_pkl,
            topic=args.topic,
        )
        pretty_print_json(output_json)
    else:
        output_json = (
            "output/perturbation_performance_"
            + args.corpora[0]
            + "_"
            + args.topic
            + ".json"
        )
        output_pkl = (
            "output/perturbation_predictions_"
            + args.corpora[0]
            + "_"
            + args.topic
            + ".pkl"
        )
        for i, p in enumerate(perturbations):
            print("Perturbation %s of %s: %s" % (i + 1, len(perturbations), p))
            evaluate_perturbation(
                be, df, p, outputfile=output_json, predictions_file=output_pkl
            )
        pretty_print_json(output_json)
