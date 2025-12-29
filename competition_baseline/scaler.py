from sklearn.preprocessing import StandardScaler

class FeatureNormalizer:
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.fitted = False

    def fit(self, X, y):
        """
        X: (N, D)
        y: (N, 1)
        """
        self.x_scaler.fit(X)
        self.y_scaler.fit(y)
        self.fitted = True

    def transform_X(self, X):
        assert self.fitted
        return self.x_scaler.transform(X)

    def transform_y(self, y):
        assert self.fitted
        return self.y_scaler.transform(y)

    def inverse_y(self, y_norm):
        return self.y_scaler.inverse_transform(y_norm)

class SequenceNormalizer:
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.fitted = False

    def fit(self, X, y):
        """
        X: (N, T, F)
        y: (N, 1)
        """
        N, T, F = X.shape
        X_flat = X.reshape(N * T, F)

        self.x_scaler.fit(X_flat)
        self.y_scaler.fit(y)

        self.fitted = True

    def transform(self, X, y=None):
        N, T, F = X.shape
        X_flat = X.reshape(N * T, F)
        X_scaled = self.x_scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(N, T, F)

        if y is not None:
            y = self.y_scaler.transform(y)

        return X_scaled, y

    def inverse_y(self, y_scaled):
        return self.y_scaler.inverse_transform(y_scaled)