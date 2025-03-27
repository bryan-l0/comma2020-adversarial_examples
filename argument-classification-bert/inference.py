"""
Runs a pre-trained BERT model for argument classification.

You can download pre-trained models here: https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip

The model 'bert_output/ukp/bert-base-topic-sentence/all_ukp_data/' was trained on all eight topics (abortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms) from the Stab et al. corpus  (UKP Sentential Argument
Mining Corpus)

Usage: python inference.py

"""

from transformers import BertForSequenceClassification
from transformers import BertTokenizer


#from pytorch_pretrained_bert.modeling import BertForSequenceClassification
#from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np

from train import InputExample, convert_examples_to_features


num_labels = 3
model_path = 'bert_output/argument_classification_ukp_all_data/'
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



tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
eval_features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, output_attentions=True)
model.to(device)
model.eval()

predicted_labels = []


with torch.no_grad():
    for input_ids, input_mask, segment_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)


        outputs = model(input_ids, segment_ids, input_mask)
        attention = outputs[1]

        example_id = 0
        head_num = -1

        last_layer = attention[-1]

        example_tokens = input_ids[example_id]
        example_attention = last_layer[example_id]

        head = example_attention[head_num]


        for i,x in enumerate(head):

            #TODO get max from row
            row = x.numpy()
            print(row)
            max = np.argmax(x.numpy())
            print(max)
            print(x[30])

            for j ,y in enumerate(x):


                if y == 0:
                    continue
                else:
                    source = int(example_tokens[i])
                    target = int(example_tokens[j])

                    source = tokenizer.convert_ids_to_tokens(source)
                    target = tokenizer.convert_ids_to_tokens(target)

                    if source == "[PAD]" or source == "[SEP]" or source == "[CLS]":
                        continue
                    #print(source, target, y)


        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        for prediction in np.argmax(logits, axis=1):
            predicted_labels.append(label_list[prediction])

print("Predicted labels:")
for idx in range(len(input_examples)):
    example = input_examples[idx]
    print("Topic:", example.text_a)
    print("Sentence:", example.text_b)
    print("Gold label:", example.label)
    print("Predicted label:", predicted_labels[idx])
    print("")

