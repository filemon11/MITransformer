from typing import Mapping

TAGSET = None  # "universal"  # None

CONTENT_POS: dict[str | None, set[str]] = dict()
PUNCTUATION: dict[str | None, set[str]] = dict()
MERGE_MAPPING: dict[str | None, dict[str, str]] = dict()

CONTENT_POS[None] = {
    "FW", "MD", "NN", "NNS", "NNP",
    "NNPS", "VB", "VBD", "VBG", "VBN",
    "VBP", "VBZ"}
PUNCTUATION[None] = {"''", "(", "SYM", "POS"}
MERGE_MAPPING[None] = {
        "JJS": "JJ", "JJR": "JJ",
        "PRP$": "PRP",
        "WP$": "PRP", "WP": "PRP", "WRB": "RB", "WDT": "DT",
        "VBD": "VB", "VBG": "VB", "VBN": "VB",
        "VBP": "VB", "VBZ": "VB",
        "PDT": "DT",
        "RBR": "RB", "RBS": "RB",
        "NNS": "NN", "NNPS": "NN", "NNP": "NN",
        "PRP$": "PRP",
        "UH": "RP",
        "SYM": "NN"}
# in natural stories SYM is assigned e.g. to thirty-two in 1632

CONTENT_POS["universal"] = {"NOUN", "VERB"}  # , "JJ", "JJR", "JJS"}
PUNCTUATION["universal"] = {"."}
MERGE_MAPPING["universal"] = {}


DEPREL_MERGE_MAPPING = {
    "nsubj": "cpreddep", "nsubjpass": "cpreddep", "dobj": "cpreddep",
    "iobj": "cpreddep",
    "csubj": "cpreddep", "csubjpass": "cpreddep", "ccomp": "cpreddep",
    "xcomp": "cpreddep",
    "dative": "cpreddep", "attr": "cpreddep", "acomp": "cpreddep",
    "nummod": "noundep", "appos": "noundep", "nmod": "noundep",
    "oprd": "noundep",
    "acl": "noundep",
    "amod": "noundep", "det": "noundep", "predet": "noundep",
    "advcl": "npreddep",
    "advmod": "npreddep", "npadvmod": "npreddep",
    "agent": "npreddep", "relcl": "npreddep", "neg": "npreddep",
    "vocative": "spcldep", "discourse": "spcldep", "expl": "spcldep",
    "aux": "spcldep", "auxpass": "spcldep", "cop": "spcldep",
    "punct": "spcldep",
    "mark": "spcldep",
    "prt": "spcldep",
    "compound": "comp", "name": "comp",
    "mwe": "comp", "foreign": "comp",
    "goeswith": "comp",
    "conj": "coord", "cc": "coord", "preconj": "coord",
    "case": "case", "pcomp": "case", "pobj": "cpreddep", "poss": "npreddep",
    "prep": "case",
    "list": "loose", "dislocated": "loose", "parataxis": "loose",
    "remnant": "loose", "reparandum": "loose",
    "root": "other", "dep": "other", "intj": "other", "quantmod": "other"
}


def merge(
        tag: str, mapping: Mapping[str, str]
        ) -> str:
    if tag not in mapping.keys():
        return tag
    else:
        return mapping[tag]


def pos_merge(
        tag: str, tagset: str | None = TAGSET) -> str:
    return merge(tag, MERGE_MAPPING[tagset])


def deprel_merge(tag: str) -> str:
    return merge(tag, DEPREL_MERGE_MAPPING)
