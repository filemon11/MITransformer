import pandas as pd
from itertools import cycle


pred_data = pd.read_csv("Provo_Corpus-Predictability_Norms.csv", encoding="ISO-8859-1")
rt_data = pd.read_csv("Provo_Corpus-Eyetracking_Data.csv", encoding="ISO-8859-1")

pred_data_fs = pred_data.drop_duplicates("Text_ID")
real_sep = [word for sentence in pred_data_fs["Text"] for word in sentence.split()]
real_textnum = [pred_data_fs.iloc[i]["Text_ID"] for i, sentence in enumerate(pred_data_fs["Text"]) for _ in sentence.split()]

pred_data_fw = pred_data.drop_duplicates(["Text_ID", "Word_Number"])
real_wisnum = pred_data_fw[["Word_In_Sentence_Number", "Word_Number"]]

def gen_wisnum():
    iter_texts = iter(pred_data_fs["Text"])
    current_text_iter = []
    for row_i in range(len(real_wisnum)):
        if real_wisnum.iloc[row_i]["Word_Number"] == 2:
            iterlist = list(current_text_iter)
            remaining_len = len(iterlist)
            for j in range(1, remaining_len+1):
                yield last_wisnum + j
            current_text_iter = iter(next(iter_texts).split())
            next(current_text_iter)
            yield 1
            last_wisnum = 1
        next(current_text_iter)
        new_wisnum = real_wisnum.iloc[row_i]["Word_In_Sentence_Number"]
        for j in range(new_wisnum-last_wisnum-1):
            yield last_wisnum+1+j
        yield new_wisnum
        last_wisnum = new_wisnum

real_word_in_sentence_number = list(gen_wisnum())

def gen_sentence_num(wisnum, textnum):
    wis_iter = iter(wisnum)
    tn_iter = iter(textnum)
    current_wis = next(wis_iter)
    current_tn = next(tn_iter)

    current_sentnum = 1

    try:
        while True:
            next_wis = next(wis_iter)
            next_tn = next(tn_iter)
            yield current_sentnum
            if next_tn > current_tn:
                current_sentnum = 1
            elif next_wis < current_wis:
                # does not work if sentences with only one word exist
                current_sentnum += 1

            current_wis = next_wis
            current_tn = next_tn
    except StopIteration:
        yield current_sentnum

real_sentnum = list(gen_sentence_num(
        real_word_in_sentence_number,
        real_textnum))

pd.DataFrame({
    "Text_ID": real_textnum,
    "Word_In_Sentence_Number": real_word_in_sentence_number,
    "Word": real_sep,
    "Sentence_Number": real_sentnum}).to_csv("Provo_Corpus-Eyetracking_Data_Words.csv", encoding="ISO-8859-1")
