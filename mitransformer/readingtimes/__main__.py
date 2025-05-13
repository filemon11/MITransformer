import sys

from .preparation import process

from typing import cast, Literal

if __name__ == "__main__":
    # Compute probabilities for natural stories corpus
    # based on a model trianed on Wikitext_processed
    model_name = sys.argv[1]
    corpus = sys.argv[2]
    shift = int(sys.argv[3])
    only_content_words_cost = bool(int(sys.argv[4]))
    only_content_words_left = bool(int(sys.argv[5]))
    assert corpus == "naturalstories" or corpus == "zuco"
    corpus = cast(Literal["naturalstories", "zuco"], corpus)

    if corpus == "naturalstories":
        in_file = "naturalstories-master/words.tsv"
    elif corpus == "zuco":
        in_file = "zuco/training_data.csv"
    else:
        raise Exception(f"Corpus {corpus} unknown.")

    out_file = f"RT/data/words_processed_{model_name}.csv"
    mapper = "processed/Wikitext_processed/mapper"  # TODO set to processed
    process(
        in_file, out_file, model_name, mapper,
        raw=True, corpus=corpus, shift=shift,
        only_content_words_cost=only_content_words_cost,
        only_content_words_left=only_content_words_left)
