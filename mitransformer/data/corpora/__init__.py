from .naturalstories import load_natural_stories  # noqa: F401
from .zuco import load_zuco  # noqa: F401
from .frank import load_frank  # noqa: F401

from typing import Protocol


class CorpusLoader(Protocol):
    def __call__(
        self,
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None) -> tuple[
            list[str], list[int], list[int]]: ...
