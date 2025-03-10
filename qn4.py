import numpy as np

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    scaler = StandardScaler()

    standardized_data = scaler.fit_transform(data)

    print("Original Data:")
    print(data)
    
    print("\nStandardized Data:")
    print(standardized_data)
