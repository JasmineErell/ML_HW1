###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    return X, y



def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    X = np.asarray(X)
    # np.c_ will treat 1D arrays as column vectors automatically.
    X = np.c_[np.ones(X.shape[0]), X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X



def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the loss.
    ###########################################################################
    # TODO: Implement the MSE loss function.                                  #
    ###########################################################################
    n = X.shape[0]
    predictions = X.dot(theta)
    errors = predictions - y
    J = np.sum(errors ** 2) / (2 * n)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    n = X.shape[0]
    for _ in range(num_iters):
        prediction = X @ theta
        errors = prediction - y
        gradients = (X.T @ errors) / n

        theta -= eta * gradients

        loss = compute_loss(X, y, theta)

        # Stop if the loss is invalid (nan, inf, or extremely large)
        if not np.isfinite(loss) or loss > 1e10:
            break

        J_history.append(loss)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    X_T_X = X.T @ X
    X_T_y = X.T @ y
    pinv_theta = np.linalg.inv(X_T_X) @ X_T_y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than epsilon. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent with stop condition optimization algorithm.  #
    ###########################################################################
    n = X.shape[0]
    for _ in range(max_iter):
        predictions = X @ theta
        errors = predictions - y
        gradients = (X.T @ errors) / n

        theta = theta - eta * gradients

        loss = compute_loss(X, y, theta)

        # if np.any(np.isnan(theta)) or np.any(np.isinf(theta)) or np.isnan(loss) or np.isinf(loss) or loss > 1e10:
        #     break

        # Stop if the improvement is smaller than epsilon
        if J_history and abs(J_history[-1] - loss) < epsilon:
            J_history.append(loss)
            break

        J_history.append(loss)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using
    the training dataset. Maintain a python dictionary with eta as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {}  # {eta_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best eta value.
    ###########################################################################
    for eta in etas:
        theta = np.zeros(X_train.shape[1])
        theta, J_history = gradient_descent(X_train, y_train, theta, eta, iterations)
        val_loss = compute_loss(X_val, y_val, theta)
        if np.isfinite(val_loss):
            eta_dict[eta] = val_loss
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ###########################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################

    number_of_features = X_train.shape[1]
    np.random.seed(42)
    rand_theta = np.random.rand(6)

    for i in range(5):
        feature_loss_dic = {}
        for feature in range(number_of_features):
            if feature in selected_features:
                continue
            candidates = selected_features + [feature]
            X_train_feature = apply_bias_trick(X_train[:, candidates])
            X_val_feature = apply_bias_trick(X_val[:, candidates])

            # theta = np.zeros(X_train_feature.shape[1])
            initial_theta = rand_theta[:len(candidates) + 1]
            new_theta, J_history = gradient_descent_stop_condition(
                X_train_feature, y_train, initial_theta, eta=best_eta, max_iter=iterations
            )

            val_loss = compute_loss(X_val_feature, y_val, new_theta)
            feature_loss_dic[feature] = val_loss
        best_feature_index = min(feature_loss_dic, key=feature_loss_dic.get)
        selected_features.append(best_feature_index)
    return selected_features





def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    new_features = {}
    columns = df.columns

    for col in columns:
        new_features[f'{col}^2'] = df[col] ** 2

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            new_features[f'{col1}*{col2}'] = df[col1] * df[col2]

    df_poly = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly