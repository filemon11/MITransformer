"""Module for loading natural stories corpus
from a .tsv file.
"""

from .. import tokeniser


def load_natural_stories(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None
        ) -> tuple[list[str], list[int], list[int]]:
    """Load natural stories corpus from tsv file.

    Parameters
    ----------
    input_file : str
        .tsv file that contains the corpus.
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

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    words: list[str] = []
    story_ids: list[int] = []
    word_ids: list[int] = []
    with open(input_file, "r") as file:
        for line in file:
            line.strip()
            token_id, token = line.split("\t")
            token = token[:-1]

            story_id, word_id, token_num = token_id.split(".")

            if token_num == "whole":
                if make_lower:
                    token = token.lower()

                if token_mapper is not None:
                    tokens = token.split(" ")
                    token = token_mapper.decode(
                        token_mapper.encode([tokens]),
                        to_string=True)[0]

                words.append(token.replace(" ", ""))
                story_ids.append(int(story_id))
                word_ids.append(int(word_id))
    return words, story_ids, word_ids
