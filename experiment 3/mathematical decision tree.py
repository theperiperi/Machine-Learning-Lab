import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for the feature
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.value = value                  # Value if the node is a leaf

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.root = None            # Root of the decision tree

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping criteria
        if depth == self.max_depth or num_classes == 1 or num_samples < 2:
            value = self._most_common_label(y)
            return Node(value=value)

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, num_samples, num_features)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _find_best_split(self, X, y, num_samples, num_features):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                # Calculate information gain
                gain = self._information_gain(y, left_indices, right_indices)
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, left_indices, right_indices):
        parent_entropy = self._entropy(y)

        left_y = y[left_indices]
        right_y = y[right_indices]

        # Calculate weighted average of child entropies
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        num_left = len(left_y)
        num_right = len(right_y)
        num_total = num_left + num_right

        gain = parent_entropy - ((num_left / num_total) * left_entropy + (num_right / num_total) * right_entropy)
        return gain

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        return classes[most_common_index]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Create a sample dataset
X = np.array([[2.0, 1.0],
              [2.0, 2.0],
              [4.0, 5.0],
              [5.0, 8.0],
              [6.0, 9.0],
              [7.0, 5.0],
              [8.0, 2.0],
              [9.0, 1.0]])

y = np.array([0, 0, 1, 1, 1, 1, 0, 0])

# Create and fit the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Predict
new_samples = np.array([[3.0, 3.0], [7.0, 8.0]])
predictions = tree.predict(new_samples)

print("Predictions for new samples:")
for i, sample in enumerate(new_samples):
    print("Sample {}: Class {}".format(sample, predictions[i]))