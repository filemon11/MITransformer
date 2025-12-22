import pandas as pd
from pandas import _typing as pdtyping

from . import rtprep

from typing import overload

from ..utils.logmaker import getLogger, info
logger = getLogger(__name__)


def io_join(
        corpus: rtprep.Corpus, model_name: str,
        additional_name: str,
        ) -> None:
    corpus_type: rtprep.CorpusTypes = (
        "ET" if corpus in rtprep.ET_CORPORA else "SP")
    join(
        f"RT/data/{corpus}_{additional_name}_metrics.csv",
        f"RT/data/{corpus}_{additional_name}_candidates_{model_name}.csv",
        corpus_type,
        f"RT/data/{corpus}_{additional_name}_preprocessed_{model_name}.csv",
    )


@overload
def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: rtprep.CorpusTypes, output_file: None = None,
        how: pdtyping.MergeHow = "inner"
        ) -> pd.DataFrame:
    ...


@overload
def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: rtprep.CorpusTypes, output_file: str,
        how: pdtyping.MergeHow = "inner"
        ) -> None:
    ...


def join(
        candidates_file: str | pd.DataFrame, metrics_file: str | pd.DataFrame,
        corpus_type: rtprep.CorpusTypes, output_file: str | None = None,
        how: pdtyping.MergeHow = "inner"
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
    base_columns = ('item', 'zone', 'WorkerId', 'Corpus')
    # Process depending on corpus type
    if corpus_type == "ET":
        measurements = (
            measurements
            .groupby(list(base_columns), as_index=False)
            .agg({
                'FFD': 'sum',
                'GPT': 'sum',
                'RBT': 'sum',
                'GD':  'sum',
                'word': 'first'
            })
            .drop_duplicates()
        )

        interest = list(base_columns) + ['FFD', 'GPT', 'RBT', 'GD']

    else:
        measurements = (
            measurements
            .groupby(list(base_columns), as_index=False)
            .agg({
                'RT': 'sum',
                'word': 'first'
            })
            .drop_duplicates()
        )

        interest = list(base_columns) + ['RT']

    # Select relevant columns and inner join with meta
    measurement = candidates.merge(
        measurements[interest], how=how, on=['item', 'zone', 'Corpus'])

    # Get token count (equivalent to the R token check)
    token_count = (
        measurement[['item', 'zone', 'Corpus', 'word']]
        .drop_duplicates()
        .shape[0]
    )
    info(0, logger, f"Token count: {token_count}")
    if output_file is None:
        return measurement
    # Write output
    measurement.to_csv(output_file, index=False)
    return None
