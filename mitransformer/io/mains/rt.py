from ... import readingtimes
from .. import parsing
import subprocess
import tqdm
import copy

from typing import cast


def main_rt_multiple(
        arguments: "parsing.RTParserArgs",
        word_size: int) -> None:
    assert not (
        arguments.model_name.startswith("hug:") and arguments.n_runs > 1), (
        "Evaluating more than one huggingface model is not supported.")

    for n_run in tqdm.tqdm(range(arguments.n_runs), desc="Runs"):
        run_arguments = copy.copy(arguments)
        if not arguments.model_name.startswith("hug:"):
            run_arguments.model_name = f"{arguments.model_name}_{n_run}"
        run_arguments.lme = False
        parsing.args_logic(run_arguments)

        main_rt(run_arguments, word_size)

    if arguments.lme:
        readingtimes.lme(
            arguments.model_name, arguments.dataset_name,
            arguments.name, arguments.n_runs, arguments.shift,
            f'RT/results/log_{arguments.model_name}.log')


def main_rt(
        arguments: "parsing.RTParserArgs",
        world_size: int) -> None:
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
    # TODO: implement the script above in python

    if arguments.lme:
        readingtimes.lme(
            model_name, corpus,
            arguments.name, 0, arguments.shift,
            f'RT/results/log_{arguments.model_name}.log')
