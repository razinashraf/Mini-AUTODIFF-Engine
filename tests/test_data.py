import numpy as np

from mini_autodiff.data import DataLoader
from mini_autodiff.data import DataLoader, StandardScaler


def test_dataloader_batches():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])

    y = np.array([0, 1, 0, 1, 0])

    loader = DataLoader(X, y, batch_size=2)

    batches = list(loader)

    assert len(batches) == 3

    assert np.array_equal(
        batches[0][0],
        np.array([[1, 2], [3, 4]])
    )

    assert np.array_equal(
        batches[0][1],
        np.array([0, 1])
    )

    assert np.array_equal(
        batches[2][0],
        np.array([[9, 10]])
    )

    assert np.array_equal(
        batches[2][1],
        np.array([0])
    )


def test_dataloader_shuffle():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])

    y = np.array([0, 1, 0, 1, 0])

    loader = DataLoader(
        X,
        y,
        batch_size=2,
        shuffle=True
    )

    batches = list(loader)

    shuffled_X = np.concatenate([batch[0] for batch in batches])
    shuffled_y = np.concatenate([batch[1] for batch in batches])

    assert sorted(map(tuple, shuffled_X)) == sorted(map(tuple, X))
    assert sorted(shuffled_y.tolist()) == sorted(y.tolist())

def test_dataloader_seed():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])

    y = np.array([0, 1, 0, 1, 0])

    loader1 = DataLoader(
        X, y,
        batch_size=2,
        shuffle=True,
        seed=42
    )

    loader2 = DataLoader(
        X, y,
        batch_size=2,
        shuffle=True,
        seed=42
    )

    batches1 = list(loader1)
    batches2 = list(loader2)

    for batch1, batch2 in zip(batches1, batches2):
        assert np.array_equal(batch1[0], batch2[0])
        assert np.array_equal(batch1[1], batch2[1])

def test_standard_scaler():
    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0]
    ])

    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(X)

    assert np.allclose(
        np.mean(X_normalized, axis=0),
        [0.0, 0.0]
    )

    assert np.allclose(
        np.std(X_normalized, axis=0),
        [1.0, 1.0]
    )

def test_standard_scaler_uses_training_statistics():
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0]
    ])

    X_test = np.array([
        [40.0, 400.0]
    ])

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_test_normalized = scaler.transform(X_test)

    # Training mean = [20, 200]
    # Training std = [sqrt(200/3), sqrt(20000/3)]
    expected = (X_test - scaler.mean) / scaler.std

    assert np.allclose(
        X_test_normalized,
        expected
    )

def test_dataloader_with_shuffle_and_seed():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0]
    ])

    y = np.array([0, 1, 0, 1, 0, 1])

    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(X)

    loader = DataLoader(
        X_normalized,
        y,
        batch_size=2,
        shuffle=True,
        seed=42
    )

    batches = list(loader)

    # 6 samples / batch size 2 = 3 batches
    assert len(batches) == 3

    # Every sample must still be present
    reconstructed_X = np.concatenate(
        [batch[0] for batch in batches]
    )

    reconstructed_y = np.concatenate(
        [batch[1] for batch in batches]
    )

    assert np.allclose(
        np.sort(reconstructed_X, axis=0),
        np.sort(X_normalized, axis=0)
    )

    assert np.array_equal(
        np.sort(reconstructed_y),
        np.sort(y)
    )