class Neuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
    
    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 1 if weighted_sum >= self.threshold else 0

weights = list(map(float, input("Enter weights separated by spaces: ").split()))
threshold = float(input("Enter threshold value: "))
neuron = Neuron(weights, threshold)

inputs = list(map(float, input("Enter inputs separated by spaces: ").split()))
output = neuron.activate(inputs)
print("Neuron Output:", output)