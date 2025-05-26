import sys
import subprocess

from . import preparation
from . import rtprep

from typing import cast

if __name__ == "__main__":
    # Compute probabilities for natural stories corpus
    # based on a model trianed on Wikitext_processed
    model_name = sys.argv[1]
    corpus = sys.argv[2]
    shift = int(sys.argv[3])
    only_content_words_cost = bool(int(sys.argv[4]))
    only_content_words_left = bool(int(sys.argv[5]))

    corpus_to_infile: dict[preparation.Corpus, str] = {
        "naturalstories": "naturalstories-master/words.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_SP": "frank/stimuli.txt",
    }
    try:
        in_file = corpus_to_infile[corpus]  # type: ignore
    except KeyError:
        raise Exception(f"Corpus {corpus} unknown.")
    corpus = cast(preparation.Corpus, corpus)

    out_file = f"RT/data/{corpus}_candidates_{model_name}.csv"
    mapper = "processed/Wikitext_processed/mapper"  # TODO set to processed
    preparation.process(
        in_file, out_file, model_name, mapper,
        raw=True, corpus=corpus, shift=shift,
        only_content_words_cost=only_content_words_cost,
        only_content_words_left=only_content_words_left)

    corpus_to_rt_infile: dict[preparation.Corpus, str] = {
        "naturalstories": "RT/data/processed_RTs.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/eyetracking.RT.txt",
        "frank_SP": "frank/selfpacedreading.RT.txt"
    }
    rtprep.prepare_RTs(
        corpus_to_rt_infile[corpus],
        f"RT/data/{corpus}_metrics.csv",
        corpus=corpus)

    subprocess.run(["Rscript", "--vanilla", "RT/preproc.R",
                    f"RT/data/{corpus}_candidates_{model_name}.csv",
                    f"RT/data/{corpus}_metrics.csv",
                    f"RT/data/{corpus}_preprocessed_{model_name}.csv",
                    "ET" if corpus in ("frank_ET", "zuco") else "SP"])
