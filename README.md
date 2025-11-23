## MITransformer (Mask-informed Transformer)

This is the official repo the BriGap-2 paper Modelling Expectation-based and Memory-based Predictors of Human Reading Times with Syntax-guided Attention, which trains the attention heads of a small-scale transformer model to attend to syntactically dependent items in the left context, enabling the extraction of integration costs for the prediction of human reading times. Please cite the paper as follows:

```
@inproceedings{mielczarek-etal-2025-modelling,
    title = "Modelling Expectation-based and Memory-based Predictors of Human Reading Times with Syntax-guided Attention",
    author = "Mielczarek, Lukas  and
      Bernard, Timoth{\'e}e  and
      Kallmeyer, Laura  and
      Spalek, Katharina  and
      Crabb{\'e}, Benoit",
    editor = "Bernard, Timoth{\'e}e  and
      Mickus, Timothee",
    booktitle = "Proceedings of the Second Workshop on the Bridges and Gaps between Formal and Computational Linguistics (BriGap-2)",
    month = sep,
    year = "2025",
    address = {D{\"u}sseldorf, Germany},
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.brigap-1.7/",
    pages = "52--71",
    ISBN = "979-8-89176-317-3"
}
```

The paper is available on arXiv: https://aclanthology.org/2025.brigap-1.7/

For questions/concerns/bugs please contact lukas.mielczarek at uni-duesseldorf.de.

## Quick Start

Clone repository.

```
git clone git@github.com:filemon11/MITransformer.git
cd MITransformer
```

Download the pre-trained model. (TODO)

```
wget <link>
unzip <name>.zip
```

(Optional) Download evaluation data: To reproduce experiments from our submission, download the UCL corpus from [UCL](https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip).

```
wget https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip
unzip 13428_2012_313_MOESM1_ESM.zip
mv 13428_2012_313_MOESM1_ESM frank
```

Running MITransformer.

```
# Install dependencies (using conda).
conda create -n mitransformer-latest python=3.12.2
source activate mitransformer-latest

conda install numpy nltk matplotlib seaborn conllu pandas tensorboard transformers flatten_json

pip install torch einops wordfreq spacy spacy_conll ufal.chu_liu_edmonds mmap_ninja optuna

python -m spacy download en_core_web_trf


# Preparing corpus for training (Wikitext)

python -m mitransformer --first_k none --first_k_eval_test none dataprep

# Run the experiments

## Hyperparameter search

sh hyperopt_standard.sh
sh hyperopt_supervised.sh

Results can be found in the log directory.

## Training the models

sh standard.sh
sh supervised.sh

## RT evaluation

sh correlate_RT.sh <corpus> <model_name> <mode_count> <spill_over> <cost_param> <left_param> <tokeniser>

corpus: Wikitext here.
model_count: 0 or 1, should be 1 in this case. It serves to aggregate results over several model runs.
cost_param: 0 or 1, controls whether non content-words receive an establishment cost, here 0.
left_param: 0 or 1, controls whether non content-words should be taken into account when computing integration cost, here 0.
tokeniser: optional, specify path to tokeniser if not using Wikitext corpus.

Results can be found in the RT/results directory

export PYTHONPATH=$(pwd)/pytorch:$PYTHONPATH