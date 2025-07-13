from perceptron import perceptron_classification, classify, add_bias
from titanic_test_preprocessing import preprocess
import pandas as pd
import numpy as np

categories, features = preprocess("train.csv")

weights = perceptron_classification(features, categories, 0.00001, 100000)

test_features = preprocess("test.csv")
classified = classify(weights, add_bias(test_features))

df = pd.read_csv("gender_submission.csv")
real = df["Survived"].to_numpy()

print(f"Accuracy: {np.sum(classified == real) / real.size}")
