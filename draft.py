# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# %%
def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    train_images = train_images.reshape(train_images.shape[0], -1).astype('float32') / 255
    test_images = test_images.reshape(test_images.shape[0], -1).astype('float32') / 255
    
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    return (train_images, train_labels), (test_images, test_labels)

# %%
(train_images, train_labels), (test_images, test_labels) = load_data()

# %%
def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# %%
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# %%
def forward_propagation(X, parameters):
    cache = {}
    A = X.T  # Transpose to match dimensions
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]
        if l == L:
            A = softmax(Z)
        else:
            A = relu(Z)
        cache['A' + str(l)] = A
        cache['Z' + str(l)] = Z
    
    return A, cache

# %%
def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[0]
    cross_entropy_cost = -np.sum(Y.T * np.log(AL + 1e-8)) / m
    L2_regularization_cost = (lambd / (2 * m)) * sum(np.sum(np.square(parameters['W' + str(l)])) for l in range(1, len(parameters) // 2 + 1))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

# %%
def backward_propagation(parameters, cache, X, Y, lambd):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[0]
    AL = cache['A' + str(L)]
    
    # Gradient of the cost with respect to ZL
    dZL = AL - Y.T
    grads['dW' + str(L)] = (np.dot(dZL, cache['A' + str(L-1)].T) + lambd * parameters['W' + str(L)]) / m
    grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    grads['dZ' + str(L)] = dZL
    
    for l in reversed(range(1, L)):
        dA = np.dot(parameters['W' + str(l+1)].T, grads['dZ' + str(l+1)])
        dZ = dA * (cache['Z' + str(l)] > 0)
        if l == 1:
            grads['dW' + str(l)] = (np.dot(dZ, X) + lambd * parameters['W' + str(l)]) / m
        else:
            grads['dW' + str(l)] = (np.dot(dZ, cache['A' + str(l-1)].T) + lambd * parameters['W' + str(l)]) / m
        grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
        grads['dZ' + str(l)] = dZ
    
    return grads

# %%
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

# %%
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# %%
def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, lambd=0.7, mini_batch_size=64, seed=0):
    parameters = initialize_parameters(layers_dims)
    costs = []
    
    for i in range(num_iterations):
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            AL, cache = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)
            grads = backward_propagation(parameters, cache, minibatch_X, minibatch_Y, lambd)
            parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
    
    return parameters, costs

# %%
layers_dims = [28*28, 128, 64, 10]

# %%
parameters, costs = model(train_images, train_labels, layers_dims, learning_rate=0.0075, num_iterations=3000, lambd=0.7, mini_batch_size=64, seed=0)

# %%



