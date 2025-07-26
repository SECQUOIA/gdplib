import re


# Alphanumeric sort, taken from https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    """
    Attempts to convert a string to an integer. If conversion fails, returns the original string.

    Parameters
    ----------
    s : str
        The string to convert to an integer.

    Returns
    -------
    int or str
        An integer if `s` can be converted, otherwise the original string.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]

    Parameters
    ----------
    s : str
        The string to split into chunks of strings and integers.

    Returns
    -------
    list
        A list of strings and integers extracted from the input string.
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def alphanum_sorted(l):
    """
    Sort the given list in the way that humans expect.

    Parameters
    ----------
    l : list
        The list of strings to sort.

    Returns
    -------
    list
        The list sorted in human-like alphanumeric order.
    """
    return sorted(l, key=alphanum_key)
