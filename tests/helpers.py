from types import SimpleNamespace


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return [to_namespace(item) for item in value]

    return value
