import pandas as pd
from pandas import _typing as pdtyping

from .. import data

from typing import overload

from ..utils.logmaker import getLogger, info
logger = getLogger(__name__)


def io_join(
        corpus: data.RTCorpus, model_name: str,
        additional_name: str,
        ) -> None:
    corpus_type: data.RTCorpusTypes = (
        "ET" if corpus in data.ET_CORPORA else "SP")
    join(
        f"RT/data/{corpus}_{additional_name}_metrics.csv",
        f"RT/data/{corpus}_{additional_name}_candidates_{model_name}.csv",
        corpus_type,
        f"RT/data/{corpus}_{additional_name}_preprocessed_{model_name}.csv",
    )


@overload
def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: data.RTCorpusTypes, output_file: None = None,
        how: pdtyping.MergeHow = "inner",
        rank: int | None = None,
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


@overload
def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: data.RTCorpusTypes, output_file: str,
        how: pdtyping.MergeHow = "inner",
        rank: int | None = None,
        only_interest: bool = True,
        ) -> None:
    ...


def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: data.RTCorpusTypes, output_file: str | None = None,
        how: pdtyping.MergeHow = "inner",
        rank: int | None = None,
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    # Read input files
    if isinstance(candidates_file, str):
        candidates = pd.read_csv(candidates_file)
    else:
        candidates = candidates_file
    if isinstance(metrics_file, str):
        measurements = pd.read_csv(metrics_file)
    else:
        measurements = metrics_file
    # # Process depending on corpus type

    # Select relevant columns and inner join with meta
    base_columns = ('item', 'zone', 'WorkerId', 'Corpus', 'element')
    if corpus_type == "ET":
        interest = list(base_columns) + ['FFD', 'GPT', 'GD']
        measurements[interest] = measurements[interest].fillna(0)
    else:
        interest = list(base_columns) + ['RT']

    if not only_interest:
        interest.extend([
            colname for colname in (
                "Text_ID", "Word_Number", "Sentence_Number")
            if colname in measurements.columns
        ])
    measurements = measurements[interest]

    on = ['item', 'zone', 'Corpus']
    for colname in candidates.columns:
        if colname in measurements.columns and colname not in on:
            measurements = measurements.drop(colname, axis=1)

    measurements = candidates.merge(
        measurements, how=how, on=on)

    # Get token count (equivalent to the R token check)
    token_count = (
        measurements[['item', 'zone', 'Corpus', 'word']]
        .drop_duplicates()
        .shape[0])

    info(rank, logger, f"Individual token count: {token_count}")
    if output_file is None:
        return measurements
    # Write output
    measurements.to_csv(output_file, index=False)
    return None
