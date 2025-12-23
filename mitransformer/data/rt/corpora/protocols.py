import pandas as pd

from typing import Protocol, overload


class CorpusLoader(Protocol):
    def __call__(
        self,
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False) -> tuple[
            list[str], list[str], list[int]]: ...


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
            self, input_file: str, output_file: str
            ) -> None:
        ...

    @overload
    def __call__(
            self, input_file: str, output_file: None = None
            ) -> pd.DataFrame:
        ...

    def __call__(
            self,
            input_file: str,
            output_file: str | None = None) -> None | pd.DataFrame:
        ...
