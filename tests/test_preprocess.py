from src.data.load_data import load_data
from src.data.preprocess import preprocess


def test_preprocess_split_sizes():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
