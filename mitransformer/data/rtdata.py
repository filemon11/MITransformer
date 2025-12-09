from .. import readingtimes

rt_corpus_to_text_file: dict[readingtimes.Corpus, str] = {
        "naturalstories": "naturalstories-master/words.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_SP": "frank/stimuli.txt",
    }

rt_corpus_to_measurements_file: dict[readingtimes.Corpus, str] = {
        "naturalstories": "RT/data/processed_RTs.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/eyetracking.RT.txt",
        "frank_SP": "frank/selfpacedreading.RT.txt"
    }
