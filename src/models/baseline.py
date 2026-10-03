# бейзлайн: TF-IDF на словах и символьных n-граммах + логрег.
# без него цифры трансформера ничего не значат
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union


def make_pipeline(seed):
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000, sublinear_tf=True),
        # символьные n-граммы вытягивают опечатки и склейки вроде "python/go"
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120_000, sublinear_tf=True),
    )
    return Pipeline([
        ("tfidf", feats),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)),
    ])


def fit_predict(train_texts, train_y, test_texts, seed):
    pipe = make_pipeline(seed).fit(train_texts, train_y)
    return pipe.predict(test_texts)
