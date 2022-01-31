import numpy as np

class Neuron:
    def __init__(self, activation):
        self.activation = activation
        self.weights = None

    def process(self, x):
        if self.weights is None:
            self.weights = [np.random.uniform(low=-1.0, high=1.0) for _ in range(len(x))]

        return self.activation(np.dot(self.weights, x))

    def gradient_descent(self):
        pass


if __name__ == "__main__":
    from activation import relu
    n = Neuron(relu)
    x = [2, -2]
    y = n.process(x)
    print(f"Result: {y}")
    print(f"ANN Weights: {n.weights}")