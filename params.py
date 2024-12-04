from dataclasses import dataclass
import inspect

from typing import Any, Self


def dict_info(d: dict[str, Any]) -> str:
    return "\n".join(
        ["{}={}".format(*item) for item
         in d.items()])


@dataclass
class Params():
    def to_dict(self, as_str: bool = False) -> dict[str, Any]:
        _dict = self.__dict__.copy()
        if as_str:
            _dict = {key: str(value)
                     for key, value in _dict.items()}
        return _dict

    @property
    def info(self) -> str:
        return dict_info(self.to_dict(as_str=True))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        return cls(**{
            k: v for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        })
