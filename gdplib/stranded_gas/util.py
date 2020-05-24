import re


# Alphanumeric sort, taken from https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def alphanum_sorted(l):
    """ Sort the given list in the way that humans expect.
    """
    return sorted(l, key=alphanum_key)
