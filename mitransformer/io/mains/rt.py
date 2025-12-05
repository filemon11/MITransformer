from ... import readingtimes
from .. import parsing
import subprocess

from typing import cast


def main_rt(
        arguments: "parsing.RTParserArgs",
        world_size: int):
    assert world_size == 1, "Multiprocessing for RT evaluation not implemented"
    # TODO: implement multiprocessing to be able to produce psycholinguistic
    # estimations during training.
    # Compute probabilities for natural stories corpus
    # based on a model trianed on Wikitext_processed
    model_name = arguments.model_name
    corpus = arguments.dataset_name
    only_content_words_cost = arguments.only_content_words_cost
    only_content_words_left = arguments.only_content_words_left
    mapper = arguments.mapper
    try:
        mapper = arguments.mapper
        # hug:<name> loads a huggingface tokeniser
    except IndexError:
        mapper = "processed/Wikitext_processed/mapper"
    # TODO unclear; does this mean the model must have been trained
    # on Wikitext?
    # Is the mapper not a model property that can be loaded?

    corpus_to_infile: dict[readingtimes.Corpus, str] = {
        "naturalstories": "naturalstories-master/words.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_SP": "frank/stimuli.txt",
    }
    try:
        in_file = corpus_to_infile[corpus]  # type: ignore
    except KeyError:
        raise Exception(f"Corpus {corpus} unknown.")
    corpus = cast(readingtimes.Corpus, corpus)

    out_file = f"RT/data/{corpus}_{arguments.name}_candidates_{model_name}.csv"

    assert arguments.masked, (
        "--masked cannot be False. Dependencies are needed for computing"
        " costs.")
    readingtimes.process(
        in_file, out_file, model_name, mapper,
        raw=True, corpus=corpus,
        only_content_words_cost=only_content_words_cost,
        only_content_words_left=only_content_words_left,
        world_size=world_size,
        shift=arguments.shift,
        trainer_args=arguments
        )

    corpus_to_rt_infile: dict[readingtimes.Corpus, str] = {
        "naturalstories": "RT/data/processed_RTs.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/eyetracking.RT.txt",
        "frank_SP": "frank/selfpacedreading.RT.txt"
    }
    readingtimes.prepare_RTs(
        corpus_to_rt_infile[corpus],
        f"RT/data/{corpus}_{arguments.name}_metrics.csv",
        corpus=corpus)

    subprocess.run([
        "Rscript", "--vanilla", "RT/preproc.R",
        f"RT/data/{corpus}_{arguments.name}_candidates_{model_name}.csv",
        f"RT/data/{corpus}_{arguments.name}_metrics.csv",
        f"RT/data/{corpus}_{arguments.name}_preprocessed_{model_name}.csv",
        "ET" if corpus in ("frank_ET", "zuco") else "SP"])

    # Evaluation
    # TODO: introduce option to evaluate several models
    # by creating a new main function and iterating over
    # a number of models and then running the evaluation
    # only once

    # TODO change cd
    subprocess.run([
        "cd", "RT", ";",
        "Rscript", "--vanilla", "analysis_new.R",
        f"{model_name}",
        "0",
        f"{corpus}",
        f"{arguments.shift}",
        f"{arguments.name}"
        # f"> RT/results/log_${model_name}.log"
        ])
