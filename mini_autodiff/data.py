import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Prevent division by zero for constant features
        self.std = np.where(self.std == 0, 1.0, self.std) #np.where(condition, value_if_true, value_if_false)

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform.")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    
class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=False, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = np.arange(len(self.X))

        rng = np.random.default_rng(self.seed)

        if self.shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]

            yield (
                self.X[batch_indices],
                self.y[batch_indices]
            )