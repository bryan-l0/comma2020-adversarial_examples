#!/usr/bin/env bash

#source activate test

testing=True
corpus="IBM"
perturbations="ner_examples_all_at_once ner_examples no_punctuation adjectives_synonyms adjectives_sat_synonyms noun_hyponym topic_alternatives adverbs_examples speculative_examples conjunctions_examples"

model_dir="ukp_bert_repo/bert_model/argument_classification_ukp_all_data/"
#model_dir="ukp_bert_repo/bert_model/ukp_all_data_trained_with_perturbations/"
#model_dir="ukp_bert_repo/bert_model/ibm_vanilla/"
#model_dir="ukp_bert_repo/bert_model/ibm_with_perturbations/"

python evaluation.py --topic "all_data" --model_dir $model_dir --perturbations $perturbations --corpora $corpus --test_mode $testing
