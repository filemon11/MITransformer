from transformers import AutoTokenizer  # type: ignore

from .. import tokeniser


def load_frank(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None
        ) -> tuple[list[str], list[int], list[int]]:
    """Load natural stories corpus from tsv file.

    Parameters
    ----------
    input_file : str
        .txt file that contains the corpus.
    make_lower : bool, default=True
        Whether to convert the tokens to lowercase.
    token_mapper_dir : str | None, default=None
        Path to a saved `tokeniser.TokenMapper`. If provided,
        every token in the .tsv file gets encoded and
        then decoded by the `tokeniser.TokenMapper` so that
        unknown tokens get replaced by th
        `tokeniser.TokenMapper.unk_token`.

    Returns
    -------
    list[str]
        The list of all tokens.
    list[int]
        For every token the story ID it appears in.
    list[int]
        For every token, its word ID.
    """

    pretokeniser = AutoTokenizer.from_pretrained(
        "bert-base-uncased").backend_tokenizer.pre_tokenizer  # type: ignore

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    words: list[str] = []
    sentence_ids: list[int] = []
    word_ids: list[int] = []

    with open(input_file, mode="r", encoding='cp1252') as file:
        file_iter = iter(file)
        next(file_iter)
        for sentence_id, line in enumerate(file_iter, start=1):
            sentence = line.split("\t")[1]

            for word_id, word in enumerate(sentence.split(), start=1):
                if make_lower:
                    word = word.lower()

                if token_mapper is not None:
                    components = [
                        tup[0] for tup in
                        pretokeniser.pre_tokenize_str(
                            word)]
                    word = token_mapper.decode(
                        token_mapper.encode([components]),
                        to_string=True, join_with="")[0]

                words.append(word)
                sentence_ids.append(sentence_id)
                word_ids.append(word_id)

    return words, sentence_ids, word_ids
