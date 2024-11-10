import math
import pickle
from collections import defaultdict

from typing import (Iterable, Self, Hashable, TypeVar,
                    Mapping, Literal, overload)

PAD = "<p>"
EOS = "<e>"
ROOT = "<r>"
DUMMY = "<d>"
UNK = "<u>"

A = TypeVar("A", bound=Hashable)
B = TypeVar("B")


class missingdict(dict[A, B]):
    def __init__(self, default_value: B, mapping: Mapping[A, B] = dict()):
        super().__init__(mapping)
        self.default_value: B = default_value

    def __missing__(self, key):
        return self.default_value


class TokenMapper():
    __pad_id: int = 0
    __unk_id: int = 1
    __dummy_id: int = 2
    __root_id: int = 3
    __eos_id: int = 4

    def __init__(self, token2id: missingdict[str, int]):
        self.token2id: missingdict[str, int] = token2id
        self.id2word: dict[int, str]
        self.id2word = {i: token for token, i in self.token2id.items()}

    @classmethod
    def train(
            cls,
            corpus: Iterable[Iterable[str]],
            unk_token: str = UNK,
            pad_token: str = PAD,
            dummmy_token: str = DUMMY,
            root_token: str = ROOT,
            eos_token: str = EOS,
            keep_top_k: int = 50_000
            ) -> Self:

        word_freqs: dict[str, float]
        word_freqs = defaultdict(
            float,
            {st: math.inf for st in (pad_token, unk_token,
                                     dummmy_token, root_token,
                                     eos_token)})

        for sentence in corpus:
            for word in sentence:
                word_freqs[word] += 1

        token2id: dict[str, int]
        selected: list[tuple[str, float]] = sorted(
            word_freqs.items(),
            key=lambda i: i[1], reverse=True
            )[:keep_top_k]

        wordset: list[str] = [token for token, _ in selected]

        token2id = {word: word_id for word_id, word in enumerate(wordset)}

        token2id = missingdict(token2id[unk_token], token2id)

        return cls(token2id)

    @classmethod
    def load(cls, filename: str) -> Self:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def unk_id(self) -> int:
        return self.__unk_id

    @property
    def pad_id(self) -> int:
        return self.__pad_id

    @property
    def dummy_id(self) -> int:
        return self.__dummy_id

    @property
    def root_id(self) -> int:
        return self.__root_id

    @property
    def eos_id(self) -> int:
        return self.__eos_id

    @property
    def unk_token(self) -> str:
        return self.id2word[self.unk_id]

    @property
    def pad_token(self) -> str:
        return self.id2word[self.pad_id]

    @property
    def eos_token(self) -> str:
        return self.id2word[self.eos_id]

    @property
    def root_token(self) -> str:
        return self.id2word[self.root_id]

    @property
    def dummy_token(self) -> str:
        return self.id2word[self.dummy_id]

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, tokens: Iterable[Iterable[str]]) -> list[list[int]]:
        return [[self.token2id[token] for token in sentence]
                for sentence in tokens]

    def encode(self, tokens: Iterable[Iterable[str]]) -> list[list[int]]:
        return self(tokens)

    @overload
    def decode(self, ids: Iterable[Iterable[int]],
               to_string: Literal[False] = False, join_with: str = " "
               ) -> list[list[str]]:
        ...

    @overload
    def decode(self, ids: Iterable[Iterable[int]],
               to_string: Literal[True], join_with: str = " "
               ) -> list[str]:
        ...

    def decode(self, ids: Iterable[Iterable[int]],
               to_string: bool = False, join_with: str = " "
               ) -> list[list[str]] | list[str]:

        if to_string:
            return [join_with.join([self.id2word[i] for i in sentence])
                    for sentence in ids]
        else:
            return [[self.id2word[i] for i in sentence] for sentence in ids]
