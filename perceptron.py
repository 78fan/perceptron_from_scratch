import numpy as np
import matplotlib.pyplot as plt

test_features = np.array([
    [1.0, 3.0], [1.5, 2.5], [2.0, 2.0], [2.5, 1.5], [3.0, 1.0],
    [1.5, 3.5], [2.0, 3.0], [2.5, 2.5], [3.0, 2.0], [3.5, 1.5],

    [0.5, 1.0], [1.0, 0.5], [1.5, 0.0], [2.0, -0.5], [2.5, -1.0],
    [-0.5, 1.5], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.5, -0.5]
])

test_categories = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

noise = np.random.uniform(-1.5, 1.5, test_features.shape)
test_features += noise

test_features = np.hstack([test_features, np.ones((test_categories.size, 1))])


def step(labels: np.ndarray) -> np.ndarray:
    return np.where(labels >= 0, 1, 0)


def score(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return features @ weights


def classify(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return step(score(weights, features))


def perceptron_error(weights: np.ndarray,
                     features: np.ndarray,
                     categories: np.ndarray) -> int:
    predicted_categories = classify(weights, features)
    error = np.abs(categories - predicted_categories)
    return error.sum()


def gradient_descent(weights: np.ndarray,
                     features: np.ndarray,
                     categories: np.ndarray,
                     step: float) -> np.ndarray:
    predicted_categories = classify(weights, features)
    gradient = features.T @ (predicted_categories - categories)
    weights -= step * gradient


def perceptron_classification(
        features: np.ndarray,
        categories: np.ndarray,
        step: float, steps: int) -> np.ndarray:
    weights = np.random.uniform(-1, 1, size=features.shape[1])
    for s in range(steps):
        gradient_descent(weights, features, categories, step)
        error = perceptron_error(weights, features, categories)
        if error == 0:
            break
        if s % 1000 == 0:
            print(error)
    return weights


def plot_classification(weights: np.ndarray,
                        features: np.ndarray,
                        categories: np.ndarray):
    plt.figure(figsize=(10, 6))
    x = features[:, 0]
    y = features[:, 1]

    plt.scatter(x[categories == 0], y[categories == 0], color='blue')
    plt.scatter(x[categories == 1], y[categories == 1], color='red')
    line = [-weights[0]/weights[1], -weights[2]/weights[1]]
    if line is not None:
        x_min, x_max = min(x), max(x)
        y_min = line[0] * x_min + line[1]
        y_max = line[0] * x_max + line[1]
        plt.plot([x_min, x_max], [y_min, y_max], color='black', linewidth=2)
    plt.title('Point color classification', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    weights = perceptron_classification(test_features, test_categories, 0.001, 100000)
    plot_classification(weights, test_features, test_categories)
