import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

path = "boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv"
seed = 42

def stem_phrase(input_str, stemmer):
    output_str = [stemmer.stem(w) for w in input_str]

    return " ".join(output_str)


def get_fitted_classifier(train_X, train_y, mode="i"):
    if mode == "i":
        clf = LogisticRegression(penalty="none", class_weight="balanced", multi_class="multinomial",  n_jobs=6,
                                 solver="lbfgs", random_state=seed, max_iter=3000)
        clf.fit(train_X, train_y)
        return clf, None
    else:
        parameters = {
            "penalty": ["l2"],
            "C": [0.0005, 0.001, 0.005, 0.01],
            # note that C is the inverse regularization (a larger C gives the model more freedom)
            "class_weight": ["balanced"],
            "multi_class": ["multinomial"],
            "solver": ["lbfgs"],
            "random_state": [seed],
            "max_iter": [3000]
        }
        model = LogisticRegression()
        clf = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=8)
        clf.fit(train_X, train_y.values.ravel())
        return clf, clf.best_params_["C"]


def analyze_clf(mode="i"):
    raw_data = pd.read_csv(path, index_col=0, header=0, dtype=int, converters={"title": str, "summary": str},
                          engine='python').dropna(axis=0)

    class_col = ["majortopic"]
    corpus = raw_data["title"]
    corpus = [title.lower().split(" ") for title in corpus]

    ps = PorterStemmer()
    corpus = [stem_phrase(title, ps) for title in corpus]

    vocab_sizes = [100, 500, 1000, 5000, 10000, 27000]

    logs = {"bal_acc": {"x": [],
                        "y": []},

            "macro_f1": {"x": [],
                         "y": []},
            "regularization": {"x": [],
                               "y": []}
            }

    for v_size in vocab_sizes:
        vectorizer = TfidfVectorizer(max_features=v_size)

        vectorized_dataset = vectorizer.fit_transform(corpus)

        devel_X, test_X, devel_y, test_y = train_test_split(vectorized_dataset, raw_data[class_col], test_size=0.2,
                                                            stratify=raw_data[class_col])
        # 20% is a quarter of the remaining 80%
        train_X, val_X, train_y, val_y = train_test_split(devel_X, devel_y, test_size=0.25, stratify=devel_y)

        print(f"""Dataset size: {vectorized_dataset.shape},
                test size: {test_X.shape}, 
                train size: {train_X.shape},
                val df: {val_X.shape}""")

        scaler = StandardScaler(with_mean=False)
        train_X = scaler.fit_transform(train_X)

        print(f"Classifying on: {v_size} top items")
        clf, inv_reg = get_fitted_classifier(train_X, train_y, mode=mode)

        test_X = scaler.transform(test_X)
        pred_test = clf.predict(test_X)

        bal_acc = balanced_accuracy_score(test_y, pred_test)
        macro_f1 = f1_score(test_y, pred_test, average="macro")

        logs["bal_acc"]["x"].append(v_size)
        logs["bal_acc"]["y"].append(bal_acc)
        logs["macro_f1"]["x"].append(v_size)
        logs["macro_f1"]["y"].append(macro_f1)

        if mode == "ii":
            logs["regularization"]["x"].append(v_size)
            logs["regularization"]["y"].append(inv_reg)

    if mode == "ii":
        fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
        ax[0].plot(logs["bal_acc"]["x"], logs["bal_acc"]["y"], 'bo-', label="Accuracy")
        ax[0].plot(logs["macro_f1"]["x"], logs["macro_f1"]["y"], 'ro-', label="F1 score")
        ax[0].legend()
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Vocabulary size")

        ax[1].plot(logs["regularization"]["x"], logs["regularization"]["y"], 'bo-', label="Inverse L2 regularization")
        ax[1].set_xlabel("Vocabulary size")
        ax[1].set_xscale("log")
        ax[1].set_ylabel("Inverse L2 regularization")
        plt.show()
    else:
        plt.plot(logs["bal_acc"]["x"], logs["bal_acc"]["y"], 'bo-', label="Accuracy")
        plt.plot(logs["macro_f1"]["x"], logs["macro_f1"]["y"], 'ro-', label="F1 score")
        plt.xscale("log")
        plt.legend()
        plt.xlabel("Vocabulary size")
        plt.show()

    return logs


if __name__ == '__main__':
    logs_i = analyze_clf(mode="i")
    logs_ii = analyze_clf(mode="ii")
    diff_acc = [(x - y) for x, y in zip(logs_ii["bal_acc"]["y"], logs_i["bal_acc"]["y"])]
    diff_f1 = [(x - y) for x, y in zip(logs_ii["macro_f1"]["y"], logs_i["macro_f1"]["y"])]
    plt.plot(logs_i["bal_acc"]["x"], diff_acc, 'bo-', label="Difference in Accuracy")
    plt.plot(logs_i["macro_f1"]["x"], diff_f1, 'ro-', label="Difference in F1 score")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Vocabulary size")
    plt.show()
