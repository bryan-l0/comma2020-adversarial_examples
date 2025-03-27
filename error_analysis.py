import json
import pprint
from tqdm import tqdm
from collections import defaultdict
from data_handler import DataHandler
from ukp_bert_repo.inference import BertEvaluator

def pretty_print_json(json_file='perturbation_performance.json'):

    pp = pprint.PrettyPrinter()
    with open(json_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        pp.pprint(data)

def get_type_ner(entities, perturbation):
    """
    Returns the changes in the perturbation compared with the original sentence.
    """
    ner_type = ''
    for entity in entities:
        text = entity['text']
        candidates = entity['candidates']
        if perturbation[entity['start_pos']:entity['end_pos']] != text:
            ner_type = entity['type']
            break
        for c in candidates:
            end_pos = entity['start_pos'] + len(c)
            if perturbation[entity['start_pos']:end_pos] == c:
                ner_type = entity['type']
                break
    return ner_type


def get_stats_ner(original_df, predictions_df):
    """
    Get stats on which type of ner works the best
    """
    ner_type_stats = {"PER":0, "ORG":0, "LOC":0, "MISC":0}

    mask = predictions_df['perturbation'] == 'ner_examples'
    predictions_df = predictions_df[mask]
    succ_mask = predictions_df['annotation'] != predictions_df['prediction']
    predictions_df = predictions_df[succ_mask]
    mask = original_df['ner'] != {"entities":[]}
    original_df = original_df[mask]

    for index,row in tqdm(original_df.iterrows(), total=len(original_df.index)):
        sent = row['sentence']
        entities = row['ner']['entities']
        mask = predictions_df['sentence'] == sent
        tmp_df = predictions_df[mask]
        for idx, r in tmp_df.iterrows():
            perturbation = r['perturbated']
            ner_type = get_type_ner(entities, perturbation)
            if ner_type == '':
                print('ERROR')
                print(1/0)
            ner_type_stats[ner_type] += 1
    return ner_type_stats


def show_successful_attacks(df, perturbation, showall=False):
    """

    :param df:
    :param showall: display unsuccessful attacks for an example with at least one successful attack
    :return:
    """
    print("======================================================")
    mask = df['perturbation'] == perturbation
    selected_df = df[mask]
    og_sentences = [row['sentence'] for i,row in selected_df.iterrows()]
    og_sentences = set(og_sentences)
    for og_sent in list(og_sentences):
        # mask the perturbations of the og sentence
        examples = selected_df['sentence'] == og_sent
        tmp_df = selected_df[examples]
        succ_mask = tmp_df['annotation'] != tmp_df['prediction']
        unsucc_mask = tmp_df['annotation'] == tmp_df['prediction']
        # Check if this sentence has any perturbation successful
        any_good = succ_mask.any()
        if any_good:
            print("Original sentence: ")
            print(og_sent)
            # print("Topic: {}".format(row["topic"]))
            print("Successful attacks: ")
            for index, row in tmp_df[succ_mask].iterrows():
                print("      Topic: {}".format(row["topic"]))
                print("      Attack: {}".format(row['perturbated']))
            if showall:
                print("Unsuccessful attacks: ")
                for index, row in tmp_df[unsucc_mask].iterrows():
                    print("      Attack: {}".format(row['perturbated']))
            print("======================================================")

def average_json(topics):

    total = defaultdict(float)

    for topic in topics:
        json_file  = "./output_testing/perturbation_performance_UKP_" + topic + ".json"
        with open(json_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            for perturbation in data.keys():
                total[perturbation] += data[perturbation]['Perturbation efficiency']
                for label in data[perturbation]['Specific efficiencies'].keys():
                    total[perturbation+label] += data[perturbation]['Specific efficiencies'][label]['efficiency']
                    total[perturbation+label+'succ'] += data[perturbation]['Specific efficiencies'][label]['successful']
                    total[perturbation+label+'unsucc'] += data[perturbation]['Specific efficiencies'][label]['unsuccessful']

    for p in total.keys():
        total[p] = total[p]/len(topics)
    pp = pprint.PrettyPrinter()
    pp.pprint(total)

def find_adversarial_training(dh):


    import numpy as np

    model_dir="ukp_bert_repo/bert_model/ukp_all_data_trained_with_perturbations/"


    print(dh.data.keys())

    label_list = ["NoArgument", "Argument_against", "Argument_for"]
    #df = DataHandler.select_by_datasplit(dh.data, 'test')
    df = dh.data
    be = BertEvaluator()
    print(len(df.index))

    df = df.loc[df['annotation']!=df['original_pred']] # changed label on UKP all
    #df = df.loc[df['perturbation']=='ner_examples']
    dh.set_data(df)
    df = dh.get_example(100)
    y_true = np.array(df['annotation'])
    sents = np.array(df['sentence'])
    topics = np.array(df['topic'])

    be.load_model(model_dir, label_list)
    ie = be.get_examples_from_dataframe(df)
    dl_tmp = be.get_dataloader(ie)
    preds = be.get_predictions(dl_tmp)
    for i, label in enumerate(y_true):
        if preds[i] == y_true[i]:
            print(topics[i])
            print(preds[i], y_true[i])
            print(sents[i])
            print()

def test_sentences(model_dir):

    import pandas as pd

    sent = 'The campaign to make pot illegal was “ firmly rooted in prejudices against Mexican immigrants and African Americans , who were associated with marijuana use at the time . ”'
    topic = 'marijuana legalization'

    label_list = ["NoArgument", "Argument_against", "Argument_for"]
    be = BertEvaluator()

    be.load_model(model_dir, label_list)

    ie = be.get_examples_from_dataframe(pd.DataFrame({'sentence':[sent],
                                                      'topic':[topic],
                                                      'annotation':[label_list[0]]
                                                      }))
    dl_tmp = be.get_dataloader(ie)
    preds = be.get_predictions(dl_tmp)
    print(preds)

def get_examples(df, perturbation, n=1):

    dh = DataHandler()
    df = df[df[perturbation].astype(bool)]

    dh.set_data(df)
    df = dh.get_example(n)

    for index, row in df.iterrows():
        print(row["topic"])
        print(len(row["sentence"].split()), row["sentence"])
        print()
        for p in row[perturbation]:
            print(p)

if __name__ == "__main__":
    # performance_json = 'output/perturbation_performance.json'
    # pretty_print_json(performance_json)

    # perturbation_pkl = "./data/dataset_perturbations_testonly.pkl"
    # dh = DataHandler.from_pickle(perturbation_pkl)
    # print(dh.data.keys())
    # df = dh.select_by_corpora(dh.data, ["UKP"])
    # get_examples(df, "adverbs_examples")
    # results_pkl = "./output/perturbation_predictions_all_data.pkl"
    # pred_dh = DataHandler.from_pickle(results_pkl)
    # original_pkl = "./data/dataset_perturbations_all.pkl"
    # orig_dh = DataHandler.from_pickle(original_pkl)
    # orig_df = orig_dh.select_by_datasplit(orig_dh.data, 'test')
    # pred_df = pred_dh.data
    # show_successful_attacks(pred_df, 'ner_examples')
    # ner_stats = get_stats_ner(orig_df, pred_df)
    # print(ner_stats)

    #pretty_print_json("./output_testing/perturbation_performance_IBM_all_data.json")
    #pretty_print_json("./output_testing/perturbation_performance_UKP_all_data.json")
    topics = ["abortion", "cloning", "death penalty", "gun control", "marijuana legalization", "minimum wage",
              "nuclear energy", "school uniforms"]
    #average_json(topics)

    dh = DataHandler.from_pickle("./output_testing/perturbation_predictions_UKP_all_data.pkl")
    find_adversarial_training(dh)


    model_dir = "argument-classification-bert/bert_output/argument_classification_ukp_all_data"
    #test_sentences(model_dir)
    model_dir = "ukp_bert_repo/bert_model/ukp_all_data_trained_with_perturbations/"
    #test_sentences(model_dir)
