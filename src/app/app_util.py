import json
def extract_param() -> float:
    """
    Extract the value of the specified parameter for the given model.

    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.

    Returns:
    - float: Value of the specified parameter.
    """
    file_path = 'src.model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data