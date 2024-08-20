import numpy as np

def init_params(input_size, hidden_size1, hidden_size2, output_size):
    W1 = np.random.rand(hidden_size1, input_size) - 0.5
    b1 = np.random.rand(hidden_size1, 1) - 0.5
    W2 = np.random.rand(hidden_size2, hidden_size1) - 0.5
    b2 = np.random.rand(hidden_size2, 1) - 0.5
    W3 = np.random.rand(output_size, hidden_size2) - 0.5
    b3 = np.random.rand(output_size, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    num_classes = len(np.unique(Y))
    Y_min = np.min(Y)
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y - Y_min, np.arange(Y.size)] = 1
    return one_hot_Y

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    sigmoid_Z = sigmoid(Z)
    return sigmoid_Z * (1 - sigmoid_Z)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = X.shape[0]
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.dot(W3.T, dZ3) * sigmoid_deriv(Z2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def calculate_f1_score(predictions, Y):
    num_classes = len(np.unique(Y))
    f1_scores = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = np.sum((predictions == i) & (Y == i))
        false_positives = np.sum((predictions == i) & (Y != i))
        false_negatives = np.sum((predictions != i) & (Y == i))

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        f1_scores[i] = 2 * (precision * recall) / (precision + recall + 1e-8)

    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score

def calculate_loss(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets have different shapes.")

    loss = np.mean((predictions - targets) ** 2)
    return loss

def gradient_descent(X, Y, alpha, iterations, input_size, hidden_size1, hidden_size2, output_size):
    W1, b1, W2, b2, W3, b3 = init_params(input_size, hidden_size1, hidden_size2, output_size)

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)

        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            f1 = calculate_f1_score(predictions, Y)
            print("Iteration:", i, "Accuracy:", accuracy, 'f1:', f1)

    return W1, b1, W2, b2, W3, b3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A2 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train