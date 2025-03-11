from dataclasses import dataclass
import inspect

from typing import Any, Self


class Undefined():
    """A special class that represents an undefined
    value. This is useful for argument parsing to
    distinguish between None and an undefined parameter
    for instance, when loading a standard configuration
    file and giving the user the option to overwrite
    parts of that configuration."""
    pass


def is_undef(obj) -> bool:
    '''The function `is_undef` checks where an object is
    undefined.

    Parameters
    ----------
    obj
        The variable to check.

    Returns
    -------
    bool
        The function `is_undef` is checking if the input `obj`
        is an instance of the `Undefined` class or
        if it is equal to the `Undefined` object.
        It returns a boolean value indicating whether `obj` is
        undefined or not.

    '''
    return isinstance(obj, Undefined) or obj == Undefined


def dict_info(d: dict[str, Any]) -> str:
    '''The function `dict_info` takes a dictionary
    as input and returns a string with key-value pairs
    separated by newlines.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to transform to a string.

    Returns
    -------
        The function `dict_info` takes a dictionary `d`
        as input and returns a string with each key-value
        pair in the dictionary formatted as "key=value" separated by newlines.

    '''
    return "\n".join(
        [
            "{}={}".format(*item) for item
            in d.items()])


@dataclass
class Params():
    '''A data class that provides functionality for managing
    parameters.
    '''
    def to_dict(self, as_str: bool = False,
                omit_undefined: bool = True) -> dict[str, Any]:
        '''Converts an object's attributes to a dictionary,
        optionally converting
        values to strings and omitting undefined values.

        Parameters
        ----------
        as_str : bool, default=False
            If `as_str` is set to `True`,
            the values will be converted to strings
            before being added to the dictionary.
        omit_undefined : bool, default=True
            If `omit_undefined` is set to `True`, any key in
            the object's dictionary with an undefined value
            will not be included in the output dictionary.
            (See `Undefined`.)

        Returns
        -------
        dict[str, Any]
            Returns a dictionary representation of the
            object's attributes.

        '''
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
        '''The `info` function returns a dictionary representation
        of the object converted to a string.

        Returns
        -------
        str
            The `info` property is returning a string representation
            of a dictionary obtained by calling
            the `to_dict` method with the `as_str` parameter set to `True`,
            and then passing the result to
            the `dict_info` function.

        '''
        return dict_info(self.to_dict(as_str=True))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        '''The function creates an instance of `Params`
        using keyword arguments,
        with default
        values for missing arguments.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for creating an instance of the class
            using those arguments. The method filters the keyword arguments
            based on the
            parameters accepted by the constructor of the class.
            If a parameter is missing in the
            keyword arguments, it is set based on the class attributes.

        Returns
        -------
            The `from_kwargs` method is returning an instance of the class
            with the parameters passed
            as keyword arguments in `kwargs`. It filters out
            only the parameters that are valid for the
            class constructor and sets default values for
            any missing parameters based on the class
            attributes.

        '''
        params = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        }
        for k in inspect.signature(cls).parameters:
            if k not in params:
                params[k] = getattr(cls, k)
        return cls(**params)

    def update_from_kwargs(self, **kwargs: Any) -> None:
        '''The function lets you update a dataclass based on
        keyword arguments.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for updating the dataclass.
            The method filters the keyword arguments
            based on the parameters accepted by the constructor of the class.

        '''
        for k, v in kwargs.items():
            if k in inspect.signature(self.__class__).parameters:
                setattr(self, k, v)
