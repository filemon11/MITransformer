"""
Necessary: spacy_conll module
"""
from natural_stories import load_natural_stories

import spacy  # type: ignore
nlp = spacy.load("en_core_web_trf")
import en_core_web_trf  # type: ignore # noqa: E402
nlp = en_core_web_trf.load()
nlp.add_pipe("conll_formatter", last=True)

import conllu   # noqa: E402

import os   # noqa: E402

from typing import Iterator, Iterable, overload  # noqa: E402

ENCODING = "utf-8"

TSV = "./naturalstories-master/words.tsv"
PARSED = "./naturalstories-master/parses/ud/stories-aligned.conllx"


def parse(text: str):
    return nlp(text)


@overload
def save_doc_as_conllu(
        doc, location: str,
        min_len: int | None = None) -> None:
    ...


@overload
def save_doc_as_conllu(
        doc, location: None,
        min_len: int | None = None) -> str:
    ...


def save_doc_as_conllu(
        doc, location: str | None,
        min_len: int | None = None) -> None | str:
    """Files should not exist or be empty
    TODO: do not repeat the code...."""
    if location is not None:
        with open(location, "a") as f:
            for sentence in doc.sents:
                if min_len is None or len(sentence._.conll) > min_len:
                    for word in sentence._.conll:
                        token = word["FORM"]

                        num_token = token.replace('.', '')
                        num_token = num_token.replace(',', '')
                        num_token = num_token.replace('-', '')
                        if num_token.isnumeric():
                            token = '<num>'
                            word['LEMMA'] = '<num>'

                        word['FORM'] = token

                        if len(token) > 0:
                            f.write("\t".join(
                                [str(entry) for entry in word.values()]))
                            f.write("\n")

                    f.write("\n\n")
        return None
    else:
        return_str = ""
        for sentence in doc.sents:
            if min_len is None or len(sentence._.conll) > min_len:
                for word in sentence._.conll:
                    token = word["FORM"]
                    num_token = token.replace('.', '')
                    num_token = num_token.replace(',', '')
                    num_token = num_token.replace('-', '')
                    if num_token.isnumeric():
                        token = '<num>'
                        word['LEMMA'] = '<num>'
                    word['FORM'] = token
                    if len(token) > 0:
                        return_str += "\t".join(
                            [str(entry) for entry in word.values()])
                        return_str += "\n"
                return_str += "\n\n"
        return return_str


def load_conllu(file: str) -> Iterator[conllu.TokenList]:
    data_file = open(file, "r", encoding=ENCODING)
    for tokenlist in conllu.parse_incr(data_file):
        yield tokenlist


def compare(doc, tokenlists: Iterator[conllu.TokenList]) -> None:

    for spacy_sent, conllu_sent in zip(doc.sents, tokenlists):
        print("Tokenisation and heads:")
        print(" ".join(
            [f"""{str(
                tok.i-spacy_sent[0].i+1)+':'+str(tok)+':'+str(
                    tok.head.i - spacy_sent[0].i+1):<10}"""
             for tok in spacy_sent]))
        print(" ".join(
            [f"{str(tok['id'])+':'+str(tok['form'])+':'+str(tok['head']):<10}"
             for tok in conllu_sent]))

        i = input("Show next sentence? (yes, no) >>> ")
        if i == "yes" or i == "":
            continue
        else:
            break


OUTPUT_DIR = "."


def parse_natural_stories_with_spacy(
        tsv_file: str = TSV,
        output_dir: str = OUTPUT_DIR,
        output_file_name: str = "natural_stories_spacy.conllu",
        min_len: int | None = None) -> None:
    """Files should not exist or be empty"""
    tokens = load_natural_stories(tsv_file)[0]
    save_doc_as_conllu(
        parse(" ".join(tokens)),
        os.path.join(output_dir, output_file_name),
        min_len=min_len)


def parse_list_of_words_with_spacy(
        list_of_words: Iterable[str],
        min_len: int | None = None
        ) -> str:
    return save_doc_as_conllu(
        parse(" ".join(list_of_words)), location=None, min_len=min_len)


# Note: Wikitext is slighty tokenised. For instance,
# it includes whitespace before punctuation.
def parse_wikitext_with_spacy(
        output_dir: str = OUTPUT_DIR,
        output_file_name_train: str = "wikitext_spacy_train2.conllu",
        output_file_name_dev: str = "wikitext_spacy_dev.conllu",
        output_file_name_test: str = "wikitext_spacy_test.conllu"):
    """Files should not exist or be empty"""

    from datasets import (   # type: ignore # noqa: E402
        load_dataset, IterableDataset, IterableDatasetDict)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    assert (not isinstance(dataset, IterableDataset)
            and not isinstance(dataset, IterableDatasetDict))
    for filename, split in zip(
            (output_file_name_train,
             output_file_name_test,
             output_file_name_dev),
            ("train", "test", "dev")):
        batch_size = 50
        for idx in range(0, len(dataset[split]), batch_size):
            lines = remove_lines(
                dataset[split][idx:idx+batch_size]["text"])    # type: ignore
            treated = remove_at_symbols("".join(lines))
            treated = remove_newlines(treated)
            treated = make_lowercase(treated)
            save_doc_as_conllu(
                parse(treated),
                os.path.join(output_dir, filename), min_len=4)


def remove_at_symbols(text: str) -> str:
    return text.replace("@-@", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")


def remove_newlines(text: str) -> str:
    return text.replace("\n", "")


def make_lowercase(text: str) -> str:
    return text.lower()


def is_empty_line(line: str) -> bool:
    if len(line.strip()) == 0:
        return True
    else:
        return False


def is_title(line: str) -> bool:
    if line.strip()[0] == "=":
        return True
    else:
        return False


def is_short(line: str, short_limit=4) -> bool:
    if len(line.split()) <= short_limit:
        return True
    else:
        return False


def remove_lines(lines: list[str]) -> list[str]:
    new_lines: list[str] = []
    for line in lines:
        if not (is_empty_line(line) or is_title(line) or is_short(line)):
            new_lines.append(line)

    return new_lines


def check_natural_stories(tsv_file: str,
                          reparsed_file: str) -> bool:
    tsv_tokens, _, _ = load_natural_stories(tsv_file)
    tokenlists = load_conllu(reparsed_file)

    append: bool = True
    parsed_tokens: list[str] = []
    for tokenlist in tokenlists:
        for token in tokenlist:
            if append:
                parsed_tokens.append(token["form"])
            else:
                assert len(parsed_tokens) > 0
                parsed_tokens[-1] += token["form"]

            if (isinstance(token["misc"], dict)
                    and token["misc"].get("SpaceAfter", "Yes") == "No"):
                append = False
            else:
                append = True

    for tsv_tok, parsed_tok in zip(tsv_tokens, parsed_tokens):
        print(tsv_tok, parsed_tok)
        if tsv_tok != parsed_tok:
            return False

    return True
