"""
Preprocess Data
"""

from gensim.parsing import preprocessing as pp


def preprocess(dataset, stemming=False):
    print('Perform preprocessing')

    data = []
    for entry in dataset:
        t = pp.preprocess_string(entry['text'].lower(), filters=[pp.strip_tags, pp.strip_punctuation,
                                                                 pp.strip_non_alphanum, pp.strip_multiple_whitespaces,
                                                                 pp.strip_numeric, pp.remove_stopwords])
        if stemming:
            t = pp.preprocess_string(t, filters=[pp.stem_text])

        data.append(t)

    return data
