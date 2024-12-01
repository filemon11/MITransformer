from dataclasses import dataclass


@dataclass
class Params():
    def to_dict(self, as_str: bool = False):
        _dict = self.__dict__.copy()
        if as_str:
            _dict = {key: str(value)
                     for key, value in _dict.items()}
        return _dict

    @property
    def info(self):
        return "\n".join(
            ["{}={}".format(*item) for item
             in self.to_dict(as_str=True).items()])
