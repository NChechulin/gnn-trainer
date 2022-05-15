from typing import Optional, Tuple


def split_in_two(string: str) -> Tuple[str, Optional[str]]:
    tokens = string.split("/")
    if len(tokens) == 2:
        return (tokens[0], tokens[1])
    else:
        return (tokens[0], None)
