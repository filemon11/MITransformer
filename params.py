from dataclasses import dataclass
import inspect

from typing import Any, Self


class Undefined():
    pass


def is_undef(obj) -> bool:
    return isinstance(obj, Undefined) or obj == Undefined


def dict_info(d: dict[str, Any]) -> str:
    return "\n".join(
        [
            "{}={}".format(*item) for item
            in d.items()])


@dataclass
class Params():
    def to_dict(self, as_str: bool = False,
                omit_undefined: bool = True) -> dict[str, Any]:
        _dict = {
            key: value for key, value    # type: ignore
            in self.__dict__.copy().items()
            if not omit_undefined or not is_undef(value)}
        if as_str:
            _dict = {
                key: str(value)
                for key, value in _dict.items()
                if not omit_undefined or not is_undef(value)}
        return _dict

    @property
    def info(self) -> str:
        return dict_info(self.to_dict(as_str=True))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        params = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        }
        for k in inspect.signature(cls).parameters:
            if k not in params:
                params[k] = getattr(cls, k)
        return cls(**params)

    def update_from_kwargs(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in inspect.signature(self.__class__).parameters:
                setattr(self, k, v)
