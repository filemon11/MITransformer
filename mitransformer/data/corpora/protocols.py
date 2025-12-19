from typing import Protocol


class CorpusLoader(Protocol):
    def __call__(
        self,
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False) -> tuple[
            list[str], list[int], list[int]]: ...


class CorpusSplitter(Protocol):
    def __call__(
        self,
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False) -> None: ...
