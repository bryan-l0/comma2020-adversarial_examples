# Generating Adversarial Examples for Topic-dependent Argument Classification

In this repository you find the code which was used for the experiments in our paper [Generating Adversarial Examples for Topic-dependent Argument Classification](https://hal.archives-ouvertes.fr/hal-02933266/document) (COMMA 2020).

In our work, we evaluate the robustness of state-of-the-art transformer models for topic-dependent argument classification by generating linguistic perturbations in the input sentences. We further improve the robustness of argument classification models using adversarial training.

# Setup
The required packages can be directly installed from the file:
```
pip install -r /path/to/requirements.txt
```

# Usage
In order to generate perturbations you must run each generation script.
To obtain the NER perturbations, you must first obtain the predictions. To do this you must run
```
python3 generate_NER_predictions.py
```
This script will add a new column to the dataset dataframe with the ner predictions.

After this you will be able to generate NER and adverbial perturbations by running:
```
python3 generate_ner_adv_perturbations.py
```
This script should generate a new file called *dataset_perturbations_all.pkl* with the original dataset plus new columns containing the perturbations generated for each type. 

Finally to get the lexical perturbations you must run:
```
python3 generate_lexical_perturbations.py
```
In the end, the file *dataset_perturbations_all.pkl* located in the directory *data/* will contain every perturbation generated.

To evaluate you must run the script *run_evaluation.sh* which should print all the results.
The models used to evaluate are not uploaded to this repository given the size of each.

# Citation

If you use our code or would like to refer to it, please cite the following paper: 
>Tobias Mayer, Santiago Marro, Elena Cabrio and Serena Villata (2020),
[Generating Adversarial Examples for Topic-dependent Argument Classification](https://hal.archives-ouvertes.fr/hal-02933266/document).
In Proceedings of the 8th International Conference on Computational Models of Argument, COMMA 2020.

```
@inproceedings{MayerCOMMA2020Adversarial,
  author    = {Tobias Mayer and
            Santiago Marro and
               Elena Cabrio and
               Serena Villata},
  title     = {Generating Adversarial Examples for Topic-dependent Argument Classification},
  series    = {Frontiers in Artificial Intelligence and Applications},
  pages ={33--44},
  volume = {326},
  booktitle = {Proceedings of the 8th International Conference on Computational Models of Argument {COMMA}},
  publisher = {{IOS} Press},
  year      = {2020},
}
```
