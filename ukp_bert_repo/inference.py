"""
Runs a pre-trained BERT model for argument classification.

You can download pre-trained models here: https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip

The model 'bert_output/ukp/bert-base-topic-sentence/all_ukp_data/' was trained on all eight topics (abortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms) from the Stab et al. corpus  (UKP Sentential Argument
Mining Corpus)

Usage: python inference.py

"""

from transformers import BertForSequenceClassification
from transformers import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.multiclass import unique_labels

from ukp_bert_repo.train import InputExample, convert_examples_to_features

class BertEvaluator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_list = None

    def load_model(self, model_path, label_list):
        self.set_tokenizer(BertTokenizer, model_path)
        self.set_model(model_path, label_list)
        self.label_list = label_list
        self.model.to(self.device)
        self.model.eval()

    def get_dataloader(self, input_examples, max_seq_length=64, batch_size=8, sampler=SequentialSampler):

        features = convert_examples_to_features(input_examples, self.label_list, max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        sampler = sampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def get_examples_from_dataframe(self, dataframe):
        input_examples = []
        for index, dfrow in dataframe.iterrows():
            input_examples.append(InputExample(text_a=dfrow['topic'], text_b=dfrow['sentence'], label=dfrow['annotation']))

        return input_examples

    def get_examples_from_perturbated_df(self, dataframe, perturbation_type):

        input_examples = []
        for index, dfrow in dataframe.iterrows():

            if perturbation_type == "topic_alternatives":
                input_examples.append(
                    InputExample(text_a=dfrow['perturbated'], text_b=dfrow['sentence'], label=dfrow['annotation']))
            else:
                input_examples.append(
                    InputExample(text_a=dfrow['topic'], text_b=dfrow['perturbated'], label=dfrow['annotation']))

        return input_examples

    def get_predictions(self, dataloader):
        predicted_labels = []
        with torch.no_grad():
            for input_ids, input_mask, segment_ids in tqdm(dataloader):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)

                outputs = self.model(input_ids, segment_ids, input_mask)

                logits = outputs[0]
                logits = logits.detach().cpu().numpy()

                for prediction in np.argmax(logits, axis=1):
                    predicted_labels.append(self.label_list[prediction])
        return predicted_labels

    def set_model(self, model_path, label_list, model_type=BertForSequenceClassification):
        num_labels = len(label_list)
        self.model = model_type.from_pretrained(model_path, num_labels=num_labels, output_attentions=True)

    def set_tokenizer(self, tokenizer_type, model_path=None, do_lower_case=True):
        if model_path is None:
            model_path =  self.model_path
        self.tokenizer = tokenizer_type.from_pretrained(model_path, do_lower_case=do_lower_case)

    def ukp_evaluate(self, dataframe):

        total_sent = 0
        correct_sent = 0
        count = {}

        y_true = []
        y_pred = []

        # replaced the prediction file reading from original authors with a pandas dataframe
        for index, dfrow in dataframe.iterrows():
            gold = dfrow['annotation']
            pred = dfrow['prediction']

            total_sent += 1
            if gold == pred:
                correct_sent += 1

            if gold not in count:
                count[gold] = {}

            if pred not in count[gold]:
                count[gold][pred] = 0

            count[gold][pred] += 1

            y_true.append(gold)
            y_pred.append(pred)
        print("gold - pred - Confusion Matrix")
        for gold_label in sorted(count.keys()):
            for pred_label in sorted(count[gold_label].keys()):
                print("%s - %s: %d" % (gold_label, pred_label, count[gold_label][pred_label]))

        print(":: BERT ::")
        print("Acc: %.2f%%" % (correct_sent / total_sent * 100))
        labels = unique_labels(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None)
        rec = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        acc = accuracy_score(y_true, y_pred)

        arg_f1 = []
        for idx, label in enumerate(labels):
            print("\n:: F1 for " + label + " ::")
            print("Prec: %.2f%%" % (prec[idx] * 100))
            print("Recall: %.2f%%" % (rec[idx] * 100))
            print("F1: %.2f%%" % (f1[idx] * 100))

            if label in labels:
                if label != 'NoArgument':
                    arg_f1.append(f1[idx])

        print("\n:: Macro Weighted for all  ::")
        print("F1: %.2f%%" % (np.mean(f1) * 100))

        prec_mapping = {key: value for key, value in zip(labels, prec)}
        rec_mapping = {key: value for key, value in zip(labels, rec)}
        return np.mean(f1), prec_mapping, rec_mapping, acc


if __name__ == '__main__':

    model_path = 'bert_model/argument_classification_ukp_all_data/'
    label_list = ["NoArgument", "Argument_against", "Argument_for"]
    max_seq_length = 64
    eval_batch_size = 8

    #Input examples. The model 'bert_output/ukp/bert-base-topic-sentence/all_topics/' expects text_a to be the topic
    #and text_b to be the sentence. label is an optional value, only used when we print the output in this script.

    input_examples = [
        InputExample(text_a='zoo', text_b='A zoo is a facility in which all animals are housed within enclosures, displayed to the public, and in which they may also breed. ', label='NoArgument'),
        InputExample(text_a='zoo', text_b='Zoos produce helpful scientific research. ', label='Argument_for'),
        InputExample(text_a='zoo', text_b='Zoos save species from extinction and other dangers.', label='Argument_for'),
        InputExample(text_a='zoo', text_b='Zoo confinement is psychologically damaging to animals.', label='Argument_against'),
        InputExample(text_a='zoo', text_b='Zoos are detrimental to animals\' physical health.', label='Argument_against'),
        InputExample(text_a='autonomous cars', text_b='Zoos are detrimental to animals\' physical health.', label='NoArgument'),
    ]

    be = BertEvaluator()
    be.load_model(model_path, label_list)
    dl = be.get_dataloader(input_examples, max_seq_length=64, batch_size=8)
    predicted_labels = be.get_predictions(dl)

    print("Predicted labels:")
    for idx in range(len(input_examples)):
        example = input_examples[idx]
        print("Topic:", example.text_a)
        print("Sentence:", example.text_b)
        print("Gold label:", example.label)
        print("Predicted label:", predicted_labels[idx])
        print("")
