"""
LDA training and evaluation
"""
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary


def train_eval(data, n_topics=10, iterations=2000, chunksize=2000, passes=1, fix_random=False):
    print('Start training')

    dictionary = Dictionary(data)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(text) for text in data]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Make a index to word dictionary.
    _ = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    if fix_random:
        random_state = 0
    else:
        random_state = np.random.RandomState()

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=n_topics,
        passes=passes,
        eval_every=None,
        random_state=random_state
    )

    print('Training finished')
    print('Prepare topics')

    top_topics = model.top_topics(texts=data, dictionary=dictionary, topn=10, coherence='c_uci')

    topics = dict()

    for k in range(len(top_topics)):
        words = []

        for w in top_topics[k][0]:
            words.append(w[1])
        topics.update({str(k): words})

    doc_topic = dict()

    for k in range(len(corpus)):
        doc_top = model.get_document_topics(corpus[k])
        li = [m[1] for m in doc_top]
        doc_topic.update({str(k): li})

    print('Calculate metrics')

    coherence, avg_topic_coherence = calc_metrics(data, len(dictionary), dictionary.token2id, top_topics, n_topics)

    metrics = dict()
    metrics.update({"coherence": coherence})
    metrics.update({"total_coherence": [avg_topic_coherence]})

    return topics, doc_topic, metrics


def calc_metrics(docs, n_terms, dictionary, top_topics, n_topics):

    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                if kk != jj:
                    dt_mat[dictionary[kk], dictionary[jj]] += 1.0

    pmi_arr = []
    for k in range(n_topics):
        top_keywords_index = [dictionary[m[1]] for m in top_topics[k][0]]
        pmi_arr.append(calculate_pmi(dt_mat, top_keywords_index))

    avg_pmi = np.average(np.array(pmi_arr))
    # print(pmi_arr)
    print('Average PMI={}'.format(avg_pmi))

    return pmi_arr, avg_pmi


def calculate_pmi(aa, top_keywords_index):
    """
    Reference:
    Short and Sparse Text Topic Modeling via Self-Aggregation
    This function has been taken over from SeaNMF implementation to provide a comparable metric
    """
    d1 = np.sum(aa)
    n_tp = len(top_keywords_index)
    pmi = []
    for index1 in top_keywords_index:
        for index2 in top_keywords_index:
            if index2 < index1:
                if aa[index1, index2] == 0:
                    pmi.append(0.0)
                else:
                    c1 = np.sum(aa[index1])
                    c2 = np.sum(aa[index2])
                    pmi.append(np.log(aa[index1, index2] * d1 / c1 / c2))
    avg_pmi = 2.0*np.sum(pmi)/float(n_tp)/(float(n_tp)-1.0)

    return avg_pmi
