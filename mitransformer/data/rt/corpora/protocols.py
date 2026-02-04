import pandas as pd

from typing import Protocol, overload


class CorpusLoader(Protocol):
    def __call__(
        self,
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False) -> tuple[
            list[str], list[str], list[str]]: ...


class CorpusSplitter(Protocol):
    def __call__(
        self,
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False) -> None: ...


class CorpusPreparer(Protocol):
    @overload
    def __call__(
            self, input_file: str, output_file: str,
            only_interest: bool = True,
            ) -> None:
        ...

    @overload
    def __call__(
            self, input_file: str, output_file: None = None,
            only_interest: bool = True,
            ) -> pd.DataFrame:
        ...

    def __call__(
            self,
            input_file: str,
            output_file: str | None = None,
            only_interest: bool = True) -> None | pd.DataFrame:
        ...
