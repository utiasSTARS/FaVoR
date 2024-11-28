#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

def print_warning(text) -> None:
    """
    Prints the provided text in orange.

    Args:
        text (str): The message to be displayed.
    """
    print("\033[93m{}\033[00m".format(text))


def print_error(text) -> None:
    """
    Prints the provided text in red.

    Args:
        text (str): The message to be displayed.
    """
    print("\033[91m{}\033[00m".format(text))


def print_success(text) -> None:
    """
    Prints the provided text in green.

    Args:
        text (str): The message to be displayed.
    """
    print("\033[92m{}\033[00m".format(text))


def print_info(text) -> None:
    """
    Prints the provided text in blue.

    Args:
        text (str): The message to be displayed.
    """
    print("\033[94m{}\033[00m".format(text))


def print_same_line(text) -> None:
    """
    Prints the provided text on the same line.

    Args:
        text (str): The message to be displayed.
    """
    print("{}".format(text), end='\r')
