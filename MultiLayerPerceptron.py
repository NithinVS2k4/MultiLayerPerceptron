import numpy as np
from PIL import Image

class Neuron:
    def __init__(self, output, weights, bias, neuron_id, layer_index):
        self.a = output
        self.w = weights
        self.b = bias
        self.neuron_id = neuron_id
        self.layer_index = layer_index  # 0 --> Input

class MLP:
    def __init__(self, layer_sizes, null_network=False):
        np.random.seed(1)

        self.layer_sizes = layer_sizes
        self.neurons = []
        self.layers = [[] for _ in range(len(layer_sizes))]
        self.weights = []
        self.biases = []

        index = -1
        for i in range(len(layer_sizes)):
            for j in range(layer_sizes[i]):
                index += 1
                if i == 0:
                    self.neurons.append(Neuron(0, [0], 0, index, i))
                else:
                    if null_network:
                        self.neurons.append(Neuron(0, np.zeros(layer_sizes[i-1]), 0, index, i))
                    else:
                        limit = np.sqrt(2 / layer_sizes[i-1]) if i < len(layer_sizes) - 1 else np.sqrt(6 / (layer_sizes[i-1] + layer_sizes[i]))
                        self.neurons.append(Neuron(0, np.random.uniform(-limit, limit, layer_sizes[i-1]), np.random.uniform(-limit, limit), index, i))
                self.layers[i].append(self.neurons[-1])

        self.weights = [[neuron.w for neuron in layer] for layer in self.layers]
        self.biases = [[neuron.b for neuron in layer] for layer in self.layers]

    def set_inputs(self, input_values, activation_function=None, log=False):
        if activation_function is None:
            activation_function = relu
        for i in range(self.layer_sizes[0]):
            self.neurons[i].a = input_values[i]
        self.forward(activation_function, log)

    def forward(self, activation_function=None, log=False):
        if activation_function is None:
            activation_function = relu
        for i in range(1, len(self.layers)):
            prev_activations = np.array([neuron.a for neuron in self.layers[i-1]])
            for neuron in self.layers[i]:
                z = np.dot(neuron.w, prev_activations) + neuron.b
                neuron.a = (sigmoid if i == len(self.layers) - 1 else activation_function)(z)
            if log:
                print(f"Layer {i} pre-activation: {z}")
                print(f"Layer {i} activation: {[neuron.a for neuron in self.layers[i]]}")

    def set_weights(self, weights, layer_id=None, neuron_id=None):
        if neuron_id is None:
            if layer_id is None:
                self.weights = weights
            else:
                self.weights[layer_id] = weights
        else:
            layer = self.neurons[neuron_id].layer_index
            self.weights[layer][neuron_id - sum(self.layer_sizes[:layer])] = weights

        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer):
                neuron.w = self.weights[i][j]

    def set_biases(self, biases, layer_id=None, neuron_id=None):
        if neuron_id is None:
            if layer_id is None:
                self.biases = biases
            else:
                self.biases[layer_id] = biases
        else:
            layer = self.neurons[neuron_id].layer_index
            self.biases[layer][neuron_id - sum(self.layer_sizes[:layer])] = biases

        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer):
                neuron.b = self.biases[i][j]

    def decrement_weights(self, weight_gradients, learning_rate):
        for i, layer in enumerate(self.layers[1:], 1):
            for j, neuron in enumerate(layer):
                neuron.w -= learning_rate * np.array(weight_gradients[i][j])
                self.weights[i][j] -= learning_rate * np.array(weight_gradients[i][j])

    def decrement_biases(self, bias_gradients, learning_rate):
        for i, layer in enumerate(self.layers[1:], 1):
            for j, neuron in enumerate(layer):
                neuron.b -= learning_rate * bias_gradients[i][j]
                self.biases[i][j] -= learning_rate * bias_gradients[i][j]

    def get_output(self):
        return [neuron.a for neuron in self.layers[-1]]

    def get_prediction(self):
        output = self.get_output()
        return output.index(max(output))

    def get_activations(self):
        return [[neuron.a for neuron in layer] for layer in self.layers]

    def input_from_image(self, image_path,size, log=False):
        image_file = Image.open(image_path)
        image_file = image_file.resize(size)
        image_file = image_file.convert('L')
        image_file.save('digit_image_resized.jpg')
        image = np.asarray(image_file)
        image = image/255
        image = image.flatten()
        self.set_inputs(image, log=log)

    def backward(self, desired_output, learning_rate, activation_function=None, act_fn_derivative=None, cost_fn_derivative=None, log = False):
        if cost_fn_derivative is None:
            cost_fn_derivative = lambda x, y: 2 * (x - y)  # Default cost derivative (x-y)**2
        if activation_function is None:
            activation_function = relu
        if act_fn_derivative is None:
            act_fn_derivative = relu_derivative

        weight_gradients = [np.zeros_like(layer_weights) for layer_weights in self.weights]
        bias_gradients = [np.zeros_like(layer_biases) for layer_biases in self.biases]

        output = self.get_output()

        delta = cost_fn_derivative(output,desired_output)*sigmoid_derivative(output)

        prev_activations = np.array([neuron.a for neuron in self.layers[-2]])

        for i in range(len(self.layers[-1])):
            weight_gradients[-1][i] += delta[i]*prev_activations
            bias_gradients[-1][i] += delta

        for i in reversed(range(1,len(self.layers)-1)):
            next_layer_weights = np.array([neuron.w for neuron in self.layers[i+1]])
            delta = (next_layer_weights.T @ delta)*np.array([act_fn_derivative(neuron.a) for neuron in self.layers[i]])

            prev_activations = np.array([neuron.a for neuron in self.layers[i-1]])
            for j, neuron in enumerate(self.layers[i]):
                weight_gradients[i][j] += delta[j] * prev_activations
                bias_gradients[i][j] += delta[j]

        self.decrement_weights(weight_gradients, learning_rate)
        self.decrement_biases(bias_gradients, learning_rate)


    def batchwise_learn(self, batch_inputs, batch_desired_outputs, learning_rate, activation_function=None, act_fn_derivative=None, cost_fn_derivative=None, log=False):
        if cost_fn_derivative is None:
            cost_fn_derivative = lambda x, y: 2 * (x - y)  # Default cost derivative (x-y)**2
        if activation_function is None:
            activation_function = relu
        if act_fn_derivative is None:
            act_fn_derivative = relu_derivative

        batch_size = len(batch_desired_outputs)
        weight_gradients = [np.zeros_like(layer_weights) for layer_weights in self.weights]
        bias_gradients = [np.zeros_like(layer_biases) for layer_biases in self.biases]

        total_loss = 0.0
        for batch_idx in range(batch_size):
            desired_output = batch_desired_outputs[batch_idx]

            # Perform forward pass
            self.set_inputs(batch_inputs[batch_idx], activation_function=activation_function, log=log)
            output = np.array(self.get_output())

            # Calculate output layer error and delta
            error = cost_fn_derivative(output, desired_output)
            delta = error * sigmoid_derivative(output)

            # Accumulate gradients for the output layer
            prev_activations = np.array([neuron.a for neuron in self.layers[-2]])
            for j, neuron in enumerate(self.layers[-1]):
                weight_gradients[-1][j] += delta[j] * prev_activations
                bias_gradients[-1][j] += delta[j]

            # Backpropagate through the hidden layers
            for i in reversed(range(1, len(self.layers) - 1)):
                next_layer_weights = np.array([neuron.w for neuron in self.layers[i+1]])
                delta = (next_layer_weights.T @ delta) * np.array([act_fn_derivative(neuron.a) for neuron in self.layers[i]])

                prev_activations = np.array([neuron.a for neuron in self.layers[i-1]])
                for j, neuron in enumerate(self.layers[i]):
                    weight_gradients[i][j] += delta[j] * prev_activations
                    bias_gradients[i][j] += delta[j]

            total_loss += np.sum((output - desired_output) ** 2)

        # Update weights and biases using accumulated gradients
        self.decrement_weights(weight_gradients, learning_rate)
        self.decrement_biases(bias_gradients, learning_rate)

        if log:
            print(self.get_activations())

        return total_loss / batch_size

    def save_parameters_text(self, filename):
        with open(filename, 'w') as file:
            for layer_weights in self.weights:
                for neuron_weights in layer_weights:
                    file.write(','.join(map(str, neuron_weights)) + '\n')
            for layer_biases in self.biases:
                for bias in layer_biases:
                    file.write(f"{bias}\n")

    def load_parameters_text(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            weight_lines = lines[:sum(len(layer) for layer in self.weights)]
            bias_lines = lines[sum(len(layer) for layer in self.weights):]

            weight_idx = 0
            for i, layer_weights in enumerate(self.weights):
                for j, _ in enumerate(layer_weights):
                    try:
                        weight_data = np.array(list(map(float, weight_lines[weight_idx].strip().split(','))))
                        self.set_weights(weight_data, neuron_id=weight_idx)
                        weight_idx += 1
                    except ValueError:
                        print(f"Error loading weights from line {weight_idx}: {weight_lines[weight_idx].strip()}")
                        continue

            bias_idx = 0
            for i, layer_biases in enumerate(self.biases):
                for j, _ in enumerate(layer_biases):
                    try:
                        bias_data = float(bias_lines[bias_idx].strip())
                        self.set_biases(bias_data, layer_id=i, neuron_id=sum(len(layer) for layer in self.layers[:i]) + j)
                        bias_idx += 1
                    except ValueError:
                        print(f"Error loading bias from line {bias_idx}: {bias_lines[bias_idx].strip()}")
                        continue

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)
