import numpy as np

class SGDRegressor:
    def __init__(self, learning_rate=0.1, max_iter=1000, tolerance=1e-3):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0
        self.errors_history = []  # Track MSE during training

    def fit(self, X, y):
        """
        Fit the model using vectorized Stochastic Gradient Descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for iteration in range(self.max_iter):
            # Compute predictions for all samples
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute errors for all samples
            errors = y_pred - y

            # Compute gradients (vectorized) with stability check
            gradient_w = np.dot(X.T, errors) / (n_samples + 1e-8)
            gradient_b = np.sum(errors) / (n_samples + 1e-8)

            # Update parameters
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            # Record Mean Squared Error for the learning curve
            mse = np.mean(errors ** 2)
            self.errors_history.append(mse)

            # Debugging: Log progress and gradient norms
            if iteration % 10 == 0:
                gradient_norm = np.linalg.norm(gradient_w)
                print(f"Iteration {iteration}, MSE = {mse}, Gradient_w Norm = {gradient_norm}, Gradient_b = {gradient_b}")

            # Convergence check
            if np.linalg.norm(gradient_w) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

    def predict(self, X):
        """
        Predict target values using the learned weights and bias.
        """
        return np.dot(X, self.weights) + self.bias
