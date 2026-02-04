from ... import readingtimes, data
from .. import parsing
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
            arguments.name, arguments.n_runs, arguments.shift)


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

    try:
        in_file = data.rt_corpus_to_text_file[corpus]  # type: ignore
    except KeyError:
        raise Exception(f"Corpus {corpus} unknown.")
    assert corpus in data.RTCORPORA
    corpus = cast(data.RTCorpus, corpus)

    assert arguments.masked, (
        "--masked cannot be False. Dependencies are needed for computing"
        " costs.")

    # Load measurements
    measurements = data.prepare_RT_measurements(
        data.rt_corpus_to_measurements_file[corpus],
        corpus=corpus, only_interest=False)

    # Load candidates
    corpus_df = readingtimes.io_corpus_convert(
        model_name, corpus, in_file, verbose=True,
        min_len=arguments.min_len_eval_test,
        max_len=arguments.max_len_eval_test)
    # Note: we are loading the RT data twice: once for
    # the lme eval and once for collecting the input of the
    # LM. We might want to unify this process

    if arguments.legacy_process:
        candidates = readingtimes.process(
            corpus_df, None, model_name, mapper,
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            world_size=world_size,
            shift=arguments.shift,
            trainer_args=arguments
        )
    else:
        assert arguments.to_add is not None
        candidates = readingtimes.new_process(
            input_file=corpus_df,
            output_file=None,
            model_dir=model_name,
            token_mapper_dir=mapper,
            to_add=arguments.to_add,
            batch_size=arguments.batch_size,
            world_size=world_size,
            use_ddp=arguments.use_ddp,
            rank=arguments.rank,
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            masked=arguments.masked,
            shift=arguments.shift,
            trainer_args=arguments,
            distr_mode=arguments.distr_mode,
            length_weighted=arguments.length_weighted,
            include_current=arguments.include_current,
            global_distr=arguments.global_distr)

    readingtimes.join(
        candidates,
        measurements,
        "ET" if corpus in data.ET_CORPORA else "SP",
        f"RT/data/{corpus}_{arguments.name}_preprocessed_{model_name}.csv",
        rank=arguments.rank,
        only_interest=False
    )

    if arguments.lme:
        readingtimes.lme(
            model_name, corpus,
            arguments.name, 0, arguments.shift)
