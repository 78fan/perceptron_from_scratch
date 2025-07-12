import numpy as np


test_features = np.array([
    [1.0, 3.0], [1.5, 2.5], [2.0, 2.0], [2.5, 1.5], [3.0, 1.0],
    [1.5, 3.5], [2.0, 3.0], [2.5, 2.5], [3.0, 2.0], [3.5, 1.5],

    [0.5, 1.0], [1.0, 0.5], [1.5, 0.0], [2.0, -0.5], [2.5, -1.0],
    [-0.5, 1.5], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.5, -0.5]
])


test_categories = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

test_features = np.hstack([test_features, np.ones((test_categories.size, 1))])

def step(labels: np.ndarray) -> np.ndarray:
    return np.where(labels >= 0, 1, 0)

def score(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return features@weights

def classify(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return step(score(weights, features))

def perceptron_error(weights: np.ndarray,
                     features: np.ndarray,
                     categories: np.ndarray) -> int:
    predicted_categories = classify(weights, features)
    error = np.abs(categories-predicted_categories)
    return error.sum()

def gradient_descent(weights: np.ndarray,
                     features: np.ndarray,
                     categories: np.ndarray,
                     step: float) -> np.ndarray:
    predicted_categories = classify(weights, features)
    gradient = features.T@(predicted_categories - categories)
    weights -= step*gradient

def perceptron_classification(
                     features: np.ndarray,
                     categories: np.ndarray,
                     step: float, steps: int) -> np.ndarray:
    weights = np.random.uniform(-1, 1, size=features.shape[1])
    for s in range(steps):
        gradient_descent(weights,features,categories,step)
        if s%1000 == 0:
            print(perceptron_error(weights, features, categories))



if __name__ == '__main__':
    perceptron_classification(test_features, test_categories, 0.00001, 100000)