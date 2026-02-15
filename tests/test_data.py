from src.data.load_data import load_data


def test_load_data_not_empty():
    X, y = load_data()
    assert len(X) > 0
    assert len(y) > 0
    assert len(X) == len(y)
