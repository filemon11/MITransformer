import math
import pickle
from collections import defaultdict

from typing import (Iterable, Self, Hashable, TypeVar,
                    Mapping, Literal, overload)

PAD = "<p>"
EOS = "<e>"
ROOT = "<r>"
DUMMY = "<d>"
UNK = "<unk>"

A = TypeVar("A", bound=Hashable)
B = TypeVar("B")


class missingdict(dict[A, B]):
    """A custom dictionary implementation that returns
    a default value for missing keys but does not
    add them to the dictionary as defaultdict does.

    Attributes
    ----------
    default_value : B
        Default value to return when given
        a key not present in the dictionary.
    """
    def __init__(self, default_value: B, mapping: Mapping[A, B] = dict()):
        '''This function initializes a `missingdict` object.

        Parameters
        ----------
        default_value : B
            The `default_value` parameter will be used if
            a key is not found in the mapping.
        mapping : Mapping[A, B], default=dict()
            The `mapping` parameter is a dictionary-like object
            that maps keys of type `A` to values of type `B`.
            The default value for this parameter is an empty dictionary.
        '''
        super().__init__(mapping)
        self.default_value: B = default_value

    def __missing__(self, _) -> B:
        '''The function `__missing__` returns the default
        value if a key is not found in the dictionary.

        Parameters
        ----------
        _
            Placeholder argument for a missing key.

        Returns
        -------
        B
            Default value (`self.default_value`).
        '''
        return self.default_value


# TODO: extend this class from missingdict
class TokenMapper():
    """This class provides methods for mapping tokens to IDs
    and vice versa, fitting the mapping based on a corpus,
    loading and saving the mapping, and encoding/decoding sequences of
    tokens.
    """
    __pad_id: int = 0
    __unk_id: int = 1
    __dummy_id: int = 2
    __root_id: int = 3
    __eos_id: int = 4

    def __init__(self, token2id: missingdict[str, int]):
        '''The function initializes a class instance with
        a dictionary mapping tokens to IDs.

        Parameters
        ----------
        token2id : missingdict[str, int]
            A dictionary that maps tokens (strings) to their corresponding IDs
            (integers).
        '''
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
        '''The `train` method takes a corpus of sentences, calculates word
        frequencies, selects the top words, and creates a mapping of tokens
        to IDs, replacing infrequent words with an "unknown" token.

        The functional tokens (unk_token, pad_token, dummy_token,
        root_token, eos_token) are always set to be part of the mapping.

        Parameters
        ----------
        corpus : Iterable[Iterable[str]]
            The `corpus` parameter in the `train` method is expected
            to be an iterable of iterables of
            strings. It represents the collection of sentences that
            will be used to train the token mapper. Each inner iterable
            should contain the words or tokens of a single sentence.
        unk_token : str
            The `unk_token` parameter is used to specify the token
            that will be used to represent unknown words in the vocabulary.
            The default value for `unk_token` is set to `UNK`. This token is
            typically used when encountering words that are not present in the
            vocabulary during
        pad_token : str
            The `pad_token` parameter is used to represent padding tokens
            in the vocabulary. These tokens are typically added to sequences
            to make them of equal length during processing.
        dummmy_token : str
            The `dummmy_token` is
            typically used to express a dependency connection with some
            upcoming token in incremental parsing.
        root_token : str
            The `root_token` is used to specify the token representing the
            root of a sentence in the corpus.
        eos_token : str
            The `eos_token` parameter represents the
            token used to indicate the end
            of a sequence or sentence.
        keep_top_k : int, optional
            The `keep_top_k` parameter in the `train` method specifies
            the maximum number of unique tokens
            to keep in the vocabulary. Only the top `keep_top_k`
            most frequent tokens will be retained in
            the vocabulary, while the rest will be replaced with
            the unknown token (UNK).

        Returns
        -------
        Self
            The method returns a trained instance of the class
            it belongs to (`Self`).

        '''

        word_freqs: dict[str, float]
        word_freqs = defaultdict(
            float,
            {st: math.inf for st in (
                pad_token, unk_token,
                dummmy_token, root_token,
                eos_token)})

        for sentence in corpus:
            for word in sentence:
                word_freqs[word] += 1

        # if num tokens < max vocab size, then definitely replace
        # three words with UNK
        token2id: dict[str, int]
        selected: list[tuple[str, float]] = sorted(
            word_freqs.items(),
            key=lambda i: i[1], reverse=True
            )[:len(word_freqs)-3][:keep_top_k]

        # replace words that appear only once with UNK
        wordset: list[str] = [token for token, freq in selected if freq > 1]

        token2id = {word: word_id for word_id, word in enumerate(wordset)}

        token2id = missingdict(token2id[unk_token], token2id)

        return cls(token2id)

    @classmethod
    def load(cls, filename: str) -> Self:
        '''The `load` class method in Python reads
        and deserializes `TokenMapper` object from a file using pickle.

        Parameters
        ----------
        filename : str
            The `filename` parameter is a string that represents the
            name of the file from which data will be loaded.

        Returns
        -------
        Self
            An instance of the `TokenMapper` class.
        '''
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    @property
    def vocab_size(self) -> int:
        '''The `vocab_size` function returns the size
        of the vocabulary based on the length of the
        `token2id` dictionary.

        Returns
        -------
        int
            The length of the
            `token2id` attribute.

        '''
        return len(self.token2id)

    @property
    def unk_id(self) -> int:
        '''The unknown token ID.

        Returns
        -------
        int
            The `unk_id` property, which is of type `int`.
        '''
        return self.__unk_id

    @property
    def pad_id(self) -> int:
        '''The padding token ID.

        Returns
        -------
        int
            The `pad_id` property, which is of type `int`.

        '''
        return self.__pad_id

    @property
    def dummy_id(self) -> int:
        '''The dummy token ID.'

        Returns
        -------
        int
            The `dummy_id` property, which is of type `int`.

        '''
        return self.__dummy_id

    @property
    def root_id(self) -> int:
        '''The root token ID.

        Returns
        -------
        int
            The `root_id` property, which is of type `int`.

        '''
        return self.__root_id

    @property
    def eos_id(self) -> int:
        '''The end of sentence token ID.

        Returns
        -------
        int
            The `eos_id` property, which is of type `int`.

        '''
        return self.__eos_id

    @property
    def unk_token(self) -> str:
        '''The unknown token.

        Returns
        -------
            The `unk_token` property is returning the
            token corresponding to the unknown token ID (`unk_id`)
            from the `id2word` dictionary.

        '''
        return self.id2word[self.unk_id]

    @property
    def pad_token(self) -> str:
        '''The padding token.

        Returns
        -------
            The `pad_token` property is returning the
            token corresponding to the unknown token ID (`pad_id`)
            from the `id2word` dictionary.

        '''
        return self.id2word[self.pad_id]

    @property
    def eos_token(self) -> str:
        '''The end of sentence token.

        Returns
        -------
            The `eos_token` property is returning the
            token corresponding to the unknown token ID (`eos_id`)
            from the `id2word` dictionary.

        '''
        return self.id2word[self.eos_id]

    @property
    def root_token(self) -> str:
        '''The root token.

        Returns
        -------
            The `root_token` property is returning the
            token corresponding to the unknown token ID (`root_id`)
            from the `id2word` dictionary.

        '''
        return self.id2word[self.root_id]

    @property
    def dummy_token(self) -> str:
        '''The dummy token.

        Returns
        -------
            The `dummy_token` property is returning the
            token corresponding to the unknown token ID (`dummy_id`)
            from the `id2word` dictionary.

        '''
        return self.id2word[self.dummy_id]

    def save(self, filename: str) -> None:
        '''The `save` function saves the object
        to a file using pickle serialization.

        Parameters
        ----------
        filename : str
            A string that represents the name of the file
            where the object will be saved using pickle.dump.

        '''
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, tokens: Iterable[Iterable[str]]) -> list[list[int]]:
        '''This function takes in a list of tokenized sentences
        and returns a list of lists where each inner list contains the
        corresponding IDs of the tokens based on a token-to-ID mapping.

        Parameters
        ----------
        tokens : Iterable[Iterable[str]]
            An iterable of iterables of strings. Each inner
            iterable represents a sentence, and
            each string within the inner iterable represents
            a token (word or symbol) in that sentence.

        Returns
        -------
            A list of lists of integers is being returned.
            Each inner list contains the integer IDs
            corresponding to the tokens in the input `tokens`
            which are looked up in the `token2id`
            dictionary.

        '''
        return [[self.token2id[token] for token in sentence]
                for sentence in tokens]

    def encode(self, tokens: Iterable[Iterable[str]]) -> list[list[int]]:
        '''This function takes in a list of tokenized sentences
        and returns a list of lists where each inner list contains the
        corresponding IDs of the tokens based on a token-to-ID mapping.

        Parameters
        ----------
        tokens : Iterable[Iterable[str]]
            An iterable of iterables of strings. Each inner
            iterable represents a sentence, and
            each string within the inner iterable represents
            a token (word or symbol) in that sentence.

        Returns
        -------
            A list of lists of integers is being returned.
            Each inner list contains the integer IDs
            corresponding to the tokens in the input `tokens`
            which are looked up in the `token2id`
            dictionary.

        '''
        return self(tokens)

    @overload
    def decode(
            self, ids: Iterable[Iterable[int]],
            to_string: Literal[False] = False, join_with: str = " "
            ) -> list[list[str]]:
        ...

    @overload
    def decode(
            self, ids: Iterable[Iterable[int]],
            to_string: Literal[True], join_with: str = " "
            ) -> list[str]:
        ...

    def decode(
            self, ids: Iterable[Iterable[int]],
            to_string: bool = False, join_with: str = " "
            ) -> list[list[str]] | list[str]:
        '''The `decode` function takes a list of lists of integers
        and converts them into a list of lists
        of strings or a list of strings based on the `to_string` parameter.

        Parameters
        ----------
        ids : Iterable[Iterable[int]]
            The `ids` parameter is an iterable containing sequences
            of integers. Each sequence represents a
            list of IDs that need to be decoded into tokens.
        to_string : bool, optional
            The `to_string` parameter in the `decode` function is a boolean
            flag that determines whether
            the output should be a list of strings or a list of lists of
            strings. If `to_string` is set to
            `True`, the function will join the decoded words in each sentence
            with the `join_with` parameter.
        join_with : str, optional
            The `join_with` parameter in the `decode` function is used
            to specify the string that will be
            used to join the decoded words into a single string when
            `to_string` is set to `True`. By default, it is set to a single
            space character " ".

        Returns
        -------
        list[list[str]] | list[str]
            The `decode` function returns a list of lists of strings
            if `to_string` is set to False, or a
            list of strings if `to_string` is set to True.

        '''

        if to_string:
            return [join_with.join([self.id2word[i] for i in sentence])
                    for sentence in ids]
        else:
            return [[self.id2word[i] for i in sentence] for sentence in ids]
