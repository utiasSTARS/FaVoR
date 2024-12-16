#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

from lib.utils_favor.log_utils import print_success, print_error, print_warning
import os
import pickle
from typing import Any


def store_obj(obj: Any, path: str) -> None:
    """
    Stores an object into a pickle file.

    Args:
        obj (Any): The object to be serialized and stored.
        path (str): The file path where the object will be saved.

    Returns:
        None
    """
    # Ensure obj is a list
    if not isinstance(obj, list):
        obj = [obj]

    # Process the object to handle standard types and custom objects
    processed_obj = [
        o if isinstance(o, (list, str, int, float, bool, type(None)))
        else o.to_dict() for o in obj
    ]

    try:
        with open(path, 'wb') as f:
            pickle.dump(processed_obj, f)
        print_success(f"Object successfully stored at: {path}")
    except Exception as e:
        print_error(f"Failed to store object: {e}")


def load_obj(path: str, ObjClass: Any, obj_name: str = "") -> Any:
    """
    Loads an object from a pickle file.

    Args:
        path (str): The file path to load the object from.
        obj_name (str, optional): A descriptive name for the object being loaded. Defaults to an empty string.

    Returns:
        Any: The loaded object, or an empty list if the file does not exist.

    Notes:
        Prints a success message if the object is loaded successfully.
        If the file does not exist, an empty list is returned.
    """
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)

                # Ensure obj is a list
                if not isinstance(obj, list):
                    obj = [obj]

                # Process the list, handling standard types and custom objects
                processed_obj = [
                    o if isinstance(o, (list, str, int, float, bool, type(None)))
                    else ObjClass.from_dict(o) for o in obj
                ]

            print_success(f"{obj_name} loaded successfully!")
            return processed_obj

        except Exception as e:
            print_error(f"Failed to load {obj_name}: {e}")
            return None
    else:
        print_warning(f"File not found: {path}. Returning an empty list.")
        return []
