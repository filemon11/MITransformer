from data import TokenMapper


def load_natural_stories(tsv_file: str,
                         make_lower: bool = True,
                         token_mapper_dir: str | None = None
                         ) -> tuple[list[str], list[int], list[int]]:
    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = TokenMapper.load(token_mapper_dir)

    words: list[str] = []
    story_ids: list[int] = []
    word_ids: list[int] = []
    with open(tsv_file, "r") as file:
        for line in file:
            line.strip()
            token_id, token = line.split("\t")
            token = token[:-1]

            story_id, word_id, token_num = token_id.split(".")

            if token_num == "whole":
                if make_lower:
                    token = token.lower()

                if token_mapper is not None:
                    tokens = token.split(" ")
                    token = token_mapper.decode(
                        token_mapper.encode([tokens]),
                        to_string=True)[0]

                words.append(token.replace(" ", ""))
                print(token)
                story_ids.append(int(story_id))
                word_ids.append(int(word_id))
    return words, story_ids, word_ids
