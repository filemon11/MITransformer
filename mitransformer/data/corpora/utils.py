import os


def create_suffixed_filepath(full_path: str, suffix: str) -> str:
    head, tail = os.path.split(full_path)
    if "." in tail:
        tail_ending, tail_name = tail[::-1].split(".", maxsplit=1)
        tail_name = tail_name[::-1]
        tail_ending = tail_ending[::-1]

        return os.path.join(head, f"{tail_name}_{suffix}.{tail_ending}")

    return os.path.join(head, f"{tail}_{suffix}")
