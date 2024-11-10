def load_natural_stories(tsv_file: str,
                         make_lower: bool = True
                         ) -> tuple[list[str], list[int], list[int]]:
    words: list[str] = []
    story_ids: list[int] = []
    word_ids: list[int] = []
    with open(tsv_file, "r") as file:
        for line in file:
            line.strip()
            token_id, token = line.split("\t")
            token = token[:-1].replace(" ", "")

            story_id, word_id, token_num = token_id.split(".")

            if token_num == "whole":
                if make_lower:
                    token = token.lower()
                words.append(token)
                story_ids.append(int(story_id))
                word_ids.append(int(word_id))

    return words, story_ids, word_ids
