import numpy as np
import pandas as pd
import os
from nltk import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def data_load():
    categories = ['neg', 'pos']
    rews = []
    class_labels = []
    for i in range(len(categories)):
        path = './review_polarity/txt_sentoken/{}'.format(categories[i])
        file_set = [f for f in os.listdir(path)]

        for file in file_set:
            with open('{}/{}'.format(path, file)) as f:
                lines = f.read().splitlines()
                rews.append(''.join(lines))
                class_labels.append(i)

    d = {'reviews': rews, 'labels': class_labels}
    df = pd.DataFrame(d)
    return df


def load_vocab():
    file_path = './opinion-lexicon-English'
    vocab = dict()

    categories = ['negative', 'positive']
    for c in categories:
        with open('{}/{}-words.txt'.format(file_path, c), encoding="ISO-8859-1") as f:
            words = f.read().splitlines()[32:]
            vocab[c] = set(words)

    return vocab


def word_count_prediction(text, lexicon):
    words = word_tokenize(text)
    pos_words = [w for w in words if w in lexicon['positive']]
    neg_words = [w for w in words if w in lexicon['negative']]
    if len(neg_words) > len(pos_words):
        return 0
    else:
        return 1


def classify_word_count(data, mode="df"):
    lexicon = load_vocab()
    if mode == "df":
        preds = data.reviews.apply(word_count_prediction, lexicon=lexicon)
    else:
        preds = data.apply(word_count_prediction, lexicon=lexicon)
    return preds


def a_i():
    data = data_load().sample(frac=1)
    train_set, test_set = train_test_split(data, test_size=0.2) # 20% of those 2000 samples = 400
    preds = classify_word_count(test_set)
    print("Accuracy:",  accuracy_score(test_set.labels, preds))
    print("F1-score: ", f1_score(test_set.labels, preds))


def a_ii():
    data = data_load().sample(frac=1)
    # 80 20 split
    train_set, test_set = train_test_split(data, test_size=0.2)
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5, ngram_range=(1, 2), max_features=1000000)
    vectorizer.fit(train_set.reviews)
    train_X = vectorizer.transform(train_set.reviews)
    train_y = train_set.labels
    test_X = vectorizer.transform(test_set.reviews)
    test_y = test_set.labels

    clf = LogisticRegression(random_state=0)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    print("Accuracy:",  accuracy_score(test_y, preds))
    print("F1-score: ", f1_score(test_y, preds))


def a_ii_bad_practice():
    data = data_load().sample(frac=1)
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5, ngram_range=(1, 2), max_features=1000000)
    x = vectorizer.fit_transform(data.reviews)
    y = data.labels
    clf = LogisticRegression()
    clf.fit(x, y)
    preds = clf.predict(x)
    print("Accuracy:",  accuracy_score(y, preds))
    print("F1-score: ", f1_score(y, preds))


def a_iii():
    data = data_load().sample(frac=1)
    vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=.5, ngram_range=(1, 2), max_features=1000000)

    num_splits = 5 # deploy, as asked on 400 samples
    k_fold_cv = KFold(n_splits=num_splits, shuffle=True)
    acc_1 = []
    acc_2 = []

    num_reps = 4
    for _ in range(num_reps):
        for idx_train, idx_test in k_fold_cv.split(data):
            train_X = data.iloc[idx_train].reviews
            train_X_v = vectorizer.fit_transform(train_X)
            test_X = data.iloc[idx_test].reviews
            test_X_v = vectorizer.transform(test_X)

            train_y = data.iloc[idx_train].labels
            test_y = data.iloc[idx_test].labels

            # word counting prediction
            preds_1 = classify_word_count(test_X, mode="col")
            acc_1.append(accuracy_score(test_y, preds_1))

            # logistic regression
            clf = LogisticRegression()
            clf.fit(train_X_v, train_y)
            preds_2 = clf.predict(test_X_v)
            acc_2.append(accuracy_score(test_y, preds_2))

    plt.hist(acc_1, alpha=0.5, label='Word counting clf')
    plt.hist(acc_2, alpha=0.5, label='Logistic regression')
    plt.legend(loc='best')
    plt.title('Classification accuracy')
    plt.xlabel('Acc. (%)')
    plt.ylabel('# of splits')
    plt.show()

    # perform a paired t test to see if the difference is significant
    pred_diff = [(p2 - p1) for p1, p2 in zip(acc_1, acc_2)]
    avg_diff = sum(pred_diff) / len(pred_diff)
    print("Average difference:", avg_diff)
    diff_std_dev = np.array(pred_diff).std()
    std_err = diff_std_dev/np.sqrt(num_splits)
    T = avg_diff / std_err
    print("t-statistic: ", T)

    n_simulations = 1000
    signs = [1, -1]
    simulated_diffs = []
    for i in range(n_simulations):
        s = list(np.random.choice(signs, num_reps * num_splits, p=[0.5, 0.5]))
        new_diffs = [x * y for x, y in zip(pred_diff, s)]
        sim_avg_diff = sum(new_diffs) / len(new_diffs)
        simulated_diffs.append(sim_avg_diff)

    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    sns.distplot(simulated_diffs, color="dodgerblue", label="Compact", **kwargs)
    plt.xlabel('Accuracy Difference')
    plt.ylabel('Count')
    plt.axvline(avg_diff, color='r', linestyle='dotted', label="Observed value", linewidth=1.5)
    plt.legend()
    plt.show()

    # compute the 0.5% quantile
    print("0.95% quantile: ", np.quantile(simulated_diffs, 0.95))


if __name__ == '__main__':
    # a_i()
    # a_ii()
    # a_ii_bad_practice()
    a_iii()
