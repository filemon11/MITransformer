from .. import readingtimes

rt_corpus_to_text_file: dict[readingtimes.Corpus, str] = {
        "naturalstories": "naturalstories-master/words.tsv",
        "naturalstories_train": "naturalstories-master/words_train.tsv",
        "naturalstories_test": "naturalstories-master/words_test.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_ET_train": "frank/stimuli_train.txt",
        "frank_ET_test": "frank/stimuli_test.txt",
        "frank_SP": "frank/stimuli.txt",
        "frank_SP_train": "frank/stimuli_train.txt",
        "frank_SP_test": "frank/stimuli_test.txt",
    }

rt_corpus_to_measurements_file: dict[readingtimes.Corpus, str] = {
        "naturalstories": "RT/data/processed_RTs.tsv",
        "naturalstories_train": "RT/data/processed_RTs.tsv",
        "naturalstories_test": "RT/data/processed_RTs.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/eyetracking.RT.txt",
        "frank_ET_train": "frank/eyetracking.RT.txt",
        "frank_ET_test": "frank/eyetracking.RT.txt",
        "frank_SP": "frank/selfpacedreading.RT.txt",
        "frank_SP_train": "frank/selfpacedreading.RT.txt",
        "frank_SP_test": "frank/selfpacedreading.RT.txt",
    }
