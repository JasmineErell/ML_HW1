import unittest
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Import the functions we want to test
from hw1 import (
    preprocess,
    apply_bias_trick,
    compute_loss,
    gradient_descent,
    compute_pinv,
    gradient_descent_stop_condition,
    find_best_learning_rate,
    forward_feature_selection,
    create_square_features
)


class LinearRegressionTests(unittest.TestCase):

    def setUp(self):
        """Set up test data that will be used across multiple tests."""
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_small = np.random.rand(5, 3)
        self.y_small = np.random.rand(5)

        self.X_zeros = np.zeros((5, 3))
        self.y_zeros = np.zeros(5)

        # Create dataset with constant feature (zero variance)
        self.X_const_feature = np.random.rand(10, 3)
        self.X_const_feature[:, 1] = 5.0  # Second column is constant
        self.y_const = np.random.rand(10)

        # Create larger dataset for more realistic testing
        self.X_large, self.y_large = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )

        # Load sample of actual data if available
        try:
            self.df_real = pd.read_csv('data.csv', nrows=100)
            if 'target' in self.df_real.columns:
                self.y_real = self.df_real['target'].values
                self.X_real = self.df_real.drop('target', axis=1).values
            else:
                # Assume last column is target
                self.y_real = self.df_real.iloc[:, -1].values
                self.X_real = self.df_real.iloc[:, :-1].values
        except Exception:
            # If data.csv not available or has issues, create backup data
            warnings.warn("Couldn't load data.csv properly, using synthetic data")
            self.X_real, self.y_real = make_regression(
                n_samples=100,
                n_features=10,
                noise=0.5,
                random_state=42
            )

    def test_preprocess_normal_case(self):
        """Test preprocessing with normal data."""
        X_proc, y_proc = preprocess(self.X_large, self.y_large)

        # Check mean is close to 0 and std close to 1
        self.assertTrue(np.allclose(np.mean(X_proc, axis=0), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.std(X_proc, axis=0), 1, atol=1e-10))
        self.assertTrue(np.isclose(np.mean(y_proc), 0, atol=1e-10))
        self.assertTrue(np.isclose(np.std(y_proc), 1, atol=1e-10))

    def test_preprocess_with_zeros(self):
        """Test preprocessing with all zeros."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore divide by zero warnings
            X_proc, y_proc = preprocess(self.X_zeros, self.y_zeros)

            # Check for NaN values that might result from division by zero
            self.assertFalse(np.isnan(X_proc).any())
            self.assertFalse(np.isnan(y_proc).any())

    def test_preprocess_constant_feature(self):
        """Test preprocessing with a constant feature (zero variance)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore divide by zero warnings
            X_proc, y_proc = preprocess(self.X_const_feature, self.y_const)

            # Check handling of constant features
            self.assertFalse(np.isnan(X_proc).any())

    def test_apply_bias_trick(self):
        """Test adding the bias column."""
        X_biased = apply_bias_trick(self.X_small)

        # Check dimensions
        self.assertEqual(X_biased.shape, (self.X_small.shape[0], self.X_small.shape[1] + 1))

        # Check first column is all ones
        self.assertTrue(np.all(X_biased[:, 0] == 1))

        # Check the rest of the data is preserved
        self.assertTrue(np.allclose(X_biased[:, 1:], self.X_small))

    def test_apply_bias_trick_with_different_types(self):
        """Test bias trick with different input types (list, dataframe)."""
        # Test with list
        X_list = self.X_small.tolist()
        X_biased = apply_bias_trick(X_list)
        self.assertEqual(X_biased.shape, (self.X_small.shape[0], self.X_small.shape[1] + 1))

        # Test with DataFrame if pandas is available
        df = pd.DataFrame(self.X_small)
        X_biased = apply_bias_trick(df)
        self.assertEqual(X_biased.shape, (self.X_small.shape[0], self.X_small.shape[1] + 1))

    def test_compute_loss_normal_case(self):
        """Test loss computation with normal data."""
        theta = np.zeros(self.X_small.shape[1] + 1)
        X_biased = apply_bias_trick(self.X_small)
        loss = compute_loss(X_biased, self.y_small, theta)

        # Loss should be positive
        self.assertGreater(loss, 0)

        # Manual calculation to verify
        n = X_biased.shape[0]
        predictions = X_biased.dot(theta)
        expected_loss = np.sum((predictions - self.y_small) ** 2) / (2 * n)
        self.assertAlmostEqual(loss, expected_loss)

    def test_compute_loss_perfect_prediction(self):
        """Test loss with perfect predictions."""
        X_biased = apply_bias_trick(self.X_small)

        # Solve for theta that gives perfect predictions
        theta = np.linalg.lstsq(X_biased, self.y_small, rcond=None)[0]
        loss = compute_loss(X_biased, self.y_small, theta)

        # Loss should be very close to 0
        self.assertAlmostEqual(loss, 0, places=10)

    def test_gradient_descent_convergence(self):
        """Test if gradient descent converges to a reasonable solution."""
        X_proc, y_proc = preprocess(self.X_large, self.y_large)
        X_biased = apply_bias_trick(X_proc)

        initial_theta = np.zeros(X_biased.shape[1])
        theta, J_history = gradient_descent(X_biased, y_proc, initial_theta, eta=0.1, num_iters=1000)

        # Check if loss decreases
        self.assertGreater(J_history[0], J_history[-1])

        # Check if solution is reasonable
        pinv_theta = compute_pinv(X_biased, y_proc)
        pinv_loss = compute_loss(X_biased, y_proc, pinv_theta)
        gd_loss = compute_loss(X_biased, y_proc, theta)

        # GD loss should be reasonably close to optimal solution
        self.assertLess(gd_loss / pinv_loss, 1.5)  # Within 50% of optimal

    def test_gradient_descent_with_high_learning_rate(self):
        """Test gradient descent with a high learning rate that might diverge."""
        X_proc, y_proc = preprocess(self.X_small, self.y_small)
        X_biased = apply_bias_trick(X_proc)

        # Try with a very high learning rate
        initial_theta = np.zeros(X_biased.shape[1])
        theta, J_history = gradient_descent(X_biased, y_proc, initial_theta, eta=10.0, num_iters=100)

        # Check if the loss history is finite (doesn't explode)
        if len(J_history) > 0:
            self.assertTrue(np.isfinite(J_history[-1]))

    def test_compute_pinv_solution(self):
        """Test if pseudoinverse gives optimal solution."""
        X_proc, y_proc = preprocess(self.X_large, self.y_large)
        X_biased = apply_bias_trick(X_proc)

        pinv_theta = compute_pinv(X_biased, y_proc)

        # Compare with numpy's lstsq which also finds the optimal solution
        optimal_theta = np.linalg.lstsq(X_biased, y_proc, rcond=None)[0]

        # Solutions should be very close
        self.assertTrue(np.allclose(pinv_theta, optimal_theta, rtol=1e-5))

    def test_compute_pinv_singular_matrix(self):
        """Test pseudoinverse with a singular matrix."""
        # Create a singular matrix (with linearly dependent columns)
        X_singular = np.zeros((5, 3))
        X_singular[:, 0] = 1  # First column all ones
        X_singular[:, 1] = 2  # Second column all twos
        X_singular[:, 2] = X_singular[:, 0] + X_singular[:, 1]  # Third is sum of first two

        X_biased = apply_bias_trick(X_singular)
        y = np.random.rand(5)

        # This should raise an exception or return NaN values
        try:
            pinv_theta = compute_pinv(X_biased, y)
            self.assertTrue(np.isnan(pinv_theta).any() or np.isinf(pinv_theta).any())
        except np.linalg.LinAlgError:
            # It's acceptable to raise an exception for singular matrices
            pass

    def test_gradient_descent_stop_condition(self):
        """Test gradient descent with stopping condition."""
        X_proc, y_proc = preprocess(self.X_large, self.y_large)
        X_biased = apply_bias_trick(X_proc)

        initial_theta = np.zeros(X_biased.shape[1])
        theta, J_history = gradient_descent_stop_condition(
            X_biased, y_proc, initial_theta, eta=0.1, max_iter=1000, epsilon=1e-8
        )

        # Check if loss decreases
        if len(J_history) > 1:
            self.assertGreater(J_history[0], J_history[-1])

        # Check if stopping condition works
        if len(J_history) > 1 and len(J_history) < 1000:
            self.assertLess(abs(J_history[-1] - J_history[-2]), 1e-8)

    def test_find_best_learning_rate(self):
        """Test finding the best learning rate."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_large, self.y_large, test_size=0.3, random_state=42
        )

        X_train_proc, y_train_proc = preprocess(X_train, y_train)
        X_val_proc, y_val_proc = preprocess(X_val, y_val)

        X_train_biased = apply_bias_trick(X_train_proc)
        X_val_biased = apply_bias_trick(X_val_proc)

        eta_dict = find_best_learning_rate(X_train_biased, y_train_proc, X_val_biased, y_val_proc, iterations=100)

        # Check if we get a dictionary with values
        self.assertIsInstance(eta_dict, dict)
        self.assertGreater(len(eta_dict), 0)

        # Check if all values are finite
        for eta, loss in eta_dict.items():
            self.assertTrue(np.isfinite(loss))

    def test_forward_feature_selection(self):
        """Test forward feature selection."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_real, self.y_real, test_size=0.3, random_state=42
        )

        X_train_proc, y_train_proc = preprocess(X_train, y_train)
        X_val_proc, y_val_proc = preprocess(X_val, y_val)

        # Find a reasonable eta first
        X_train_biased = apply_bias_trick(X_train_proc)
        X_val_biased = apply_bias_trick(X_val_proc)

        eta_dict = find_best_learning_rate(X_train_biased, y_train_proc, X_val_biased, y_val_proc, iterations=50)
        if eta_dict:
            best_eta = min(eta_dict, key=eta_dict.get)
        else:
            best_eta = 0.01  # Default if finding fails

        # Test feature selection with small number of iterations
        selected_features = forward_feature_selection(
            X_train_proc, y_train_proc, X_val_proc, y_val_proc, best_eta, iterations=50
        )

        # Check if we get 5 distinct features
        self.assertEqual(len(selected_features), 5)
        self.assertEqual(len(set(selected_features)), 5)  # No duplicates

        # All selected indices should be valid
        for idx in selected_features:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, X_train_proc.shape[1])

    def test_create_square_features(self):
        """Test creation of polynomial features."""
        # Create a small dataframe for testing
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        df_poly = create_square_features(df)

        # Check dimensions
        expected_cols = 3 + 3 + 3  # original + squared + interactions
        self.assertEqual(df_poly.shape[1], expected_cols)

        # Check squared features exist
        for col in df.columns:
            self.assertIn(f'{col}^2', df_poly.columns)
            self.assertTrue(np.allclose(df_poly[f'{col}^2'], df[col] ** 2))

        # Check interaction features exist
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if j > i:
                    self.assertIn(f'{col1}*{col2}', df_poly.columns)
                    self.assertTrue(np.allclose(df_poly[f'{col1}*{col2}'], df[col1] * df[col2]))


class EdgeCaseTests(unittest.TestCase):
    """Tests focusing specifically on edge cases."""

    def test_empty_input(self):
        """Test behavior with empty input arrays."""
        # Empty arrays (0 samples)
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])

        with self.assertRaises((ValueError, IndexError, ZeroDivisionError)):
            # One of these errors is expected
            preprocess(X_empty, y_empty)

    def test_single_sample(self):
        """Test with just one sample."""
        X_single = np.random.rand(1, 3)
        y_single = np.random.rand(1)

        # Preprocess might give NaN due to std of singleton array being 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X_proc, y_proc = preprocess(X_single, y_single)
                # If it runs, check for NaN
                self.assertTrue(np.isnan(X_proc).any() or np.isnan(y_proc).any())
            except:
                pass  # Exception is also acceptable

        # Bias trick should work normally
        X_biased = apply_bias_trick(X_single)
        self.assertEqual(X_biased.shape, (1, 4))  # 1 sample, 3 features + bias

    def test_tiny_learning_rate(self):
        """Test with extremely small learning rate."""
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        X_biased = apply_bias_trick(X)

        theta = np.zeros(X_biased.shape[1])
        theta, J_history = gradient_descent(X_biased, y, theta, eta=1e-15, num_iters=10)

        # Loss should barely change with tiny learning rate
        if len(J_history) > 1:
            self.assertAlmostEqual(J_history[0], J_history[-1], places=10)

    def test_huge_learning_rate(self):
        """Test with extremely large learning rate."""
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        X_biased = apply_bias_trick(X)

        theta = np.zeros(X_biased.shape[1])
        theta, J_history = gradient_descent(X_biased, y, theta, eta=1e10, num_iters=10)

        # Check if the implementation handles divergence
        if len(J_history) > 0:
            self.assertTrue(np.isfinite(J_history[-1]) or len(J_history) < 10)

    def test_large_values(self):
        """Test with very large input values."""
        X_large_vals = np.random.rand(10, 3) * 1e10
        y_large_vals = np.random.rand(10) * 1e10

        # Preprocess should handle this fine
        X_proc, y_proc = preprocess(X_large_vals, y_large_vals)
        self.assertTrue(np.allclose(np.mean(X_proc, axis=0), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.std(X_proc, axis=0), 1, atol=1e-10))

    def test_mixed_dtypes(self):
        """Test with mixed integer and float data."""
        X_mixed = np.array([
            [1, 2.5, 3],
            [4, 5.5, 6],
            [7, 8.5, 9]
        ])
        y_mixed = np.array([10, 11, 12])

        # Apply bias trick should convert everything to float
        X_biased = apply_bias_trick(X_mixed)
        self.assertEqual(X_biased.dtype, np.float64)


class IntegrationTests(unittest.TestCase):
    """Tests that integrate multiple functions together."""

    def test_full_pipeline(self):
        """Test the entire linear regression pipeline."""
        # Generate some data
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess
        X_train_proc, y_train_proc = preprocess(X_train, y_train)
        X_test_proc, y_test_proc = preprocess(X_test, y_test)

        # Apply bias trick
        X_train_biased = apply_bias_trick(X_train_proc)
        X_test_biased = apply_bias_trick(X_test_proc)

        # Train with pseudoinverse
        pinv_theta = compute_pinv(X_train_biased, y_train_proc)

        # Train with gradient descent
        initial_theta = np.zeros(X_train_biased.shape[1])
        gd_theta, _ = gradient_descent(X_train_biased, y_train_proc, initial_theta, eta=0.1, num_iters=1000)

        # Train with gradient descent with stop condition
        gd_stop_theta, _ = gradient_descent_stop_condition(
            X_train_biased, y_train_proc, initial_theta, eta=0.1, max_iter=1000, epsilon=1e-8
        )

        # Compute test losses
        pinv_loss = compute_loss(X_test_biased, y_test_proc, pinv_theta)
        gd_loss = compute_loss(X_test_biased, y_test_proc, gd_theta)
        gd_stop_loss = compute_loss(X_test_biased, y_test_proc, gd_stop_theta)

        # All losses should be finite
        self.assertTrue(np.isfinite(pinv_loss))
        self.assertTrue(np.isfinite(gd_loss))
        self.assertTrue(np.isfinite(gd_stop_loss))

        # Pseudoinverse should give the best results
        self.assertLessEqual(pinv_loss, gd_loss * 1.1)  # Allow 10% wiggle room

    def test_square_features_improvement(self):
        """Test if square features improve performance."""
        # Generate nonlinear data
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = 1.5 * X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2 + 0.1 * X[:, 0] * X[:, 1] + 0.1 * np.random.randn(100)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create DataFrames
        df_train = pd.DataFrame(X_train, columns=['feature1', 'feature2'])
        df_test = pd.DataFrame(X_test, columns=['feature1', 'feature2'])

        # Create square features
        df_train_poly = create_square_features(df_train)
        df_test_poly = create_square_features(df_test)

        # Convert back to numpy for processing
        X_train_poly = df_train_poly.values
        X_test_poly = df_test_poly.values

        # Preprocess
        X_train_proc, y_train_proc = preprocess(X_train, y_train)
        X_train_poly_proc, _ = preprocess(X_train_poly, y_train)
        X_test_proc, y_test_proc = preprocess(X_test, y_test)
        X_test_poly_proc, _ = preprocess(X_test_poly, y_test)

        # Apply bias trick
        X_train_biased = apply_bias_trick(X_train_proc)
        X_train_poly_biased = apply_bias_trick(X_train_poly_proc)
        X_test_biased = apply_bias_trick(X_test_proc)
        X_test_poly_biased = apply_bias_trick(X_test_poly_proc)

        # Train with pseudoinverse
        theta_linear = compute_pinv(X_train_biased, y_train_proc)
        theta_poly = compute_pinv(X_train_poly_biased, y_train_proc)

        # Compute test losses
        loss_linear = compute_loss(X_test_biased, y_test_proc, theta_linear)
        loss_poly = compute_loss(X_test_poly_biased, y_test_proc, theta_poly)

        # Polynomial features should improve performance on this nonlinear data
        self.assertLess(loss_poly, loss_linear)


if __name__ == '__main__':
    unittest.main()