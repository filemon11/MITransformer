from . import functions
from .. import parsing


def main_dataprep(arguments: "parsing.ParserArgs") -> None:
    functions._load_data_provider(arguments, memmaped=False)
