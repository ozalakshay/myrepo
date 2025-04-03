(https://colab.research.google.com/drive/1mGS-eN_4Y7DqsnsD3s_NN0VH_bJcl-mu#scrollTo=5OFa31b4l5nT)
---------------------------------------

Below is a suggested “theory page” for each practical (1–10) based on the practical tasks listed in the PDF. Each page is formatted with an Aim, a brief (yet descriptive) Theory section, and a Conclusion. For practicals that include two parts (A and B), each part has its own set of sections. You can adjust or expand these summaries as needed for your exam preparation.


---

Practical No. 1

A. Design a simple linear neural network model

Aim:
To design and implement a basic linear neural network that forms the foundation for more complex models.

Theory:
A linear neural network consists of an input layer directly connected to an output layer with no hidden layers. Its computation is based on weighted sums of inputs. This simple structure helps in understanding the basic mechanics of neural processing.

Conclusion:
The exercise demonstrates the core concept of weighted linear combinations and lays the groundwork for understanding more advanced neural architectures.


B. Calculate the output of neural net using both binary and bipolar sigmoidal function

Aim:
To compute and compare the outputs of a neural network when using two different sigmoid activation functions.

Theory:
The binary sigmoidal function produces outputs between 0 and 1, while the bipolar version yields values between –1 and 1. These functions introduce non-linearity into the network and affect the way output values are interpreted.

Conclusion:
The experiment shows how the choice of activation function influences the network’s output, highlighting the importance of activation in model behavior.



---

Practical No. 2

A. Generate AND/NOT function using McCulloch-Pitts neural net

Aim:
To implement basic logical operations using the McCulloch-Pitts model.

Theory:
The McCulloch-Pitts neuron uses threshold logic to simulate binary operations. For AND/NOT functions, inputs are combined with assigned weights and compared against a threshold to produce a binary output.

Conclusion:
The results validate that simple threshold units can effectively reproduce basic logical functions.


B. Generate XOR function using McCulloch-Pitts neural net

Aim:
To attempt the implementation of the XOR function using the same model.

Theory:
Unlike AND/NOT, the XOR function is not linearly separable and cannot be represented by a single McCulloch-Pitts neuron. This limitation highlights the need for multi-layer architectures for non-linear problems.

Conclusion:
The exercise reinforces the concept that certain logical functions require more complex network structures beyond single-layer implementations.



---

Practical No. 3

A. Write a program to implement Hebb’s rule

Aim:
To apply Hebbian learning for adjusting connection strengths in a neural network.

Theory:
Hebb’s rule is based on the principle that simultaneous activation of neurons strengthens their mutual connection. This learning rule is key for understanding how associations form in neural systems.

Conclusion:
The experiment illustrates that simple weight adjustments following Hebb’s principle can encode associative memories.


B. Write a program to implement the delta rule

Aim:
To use error correction (gradient descent) to update neural network weights.

Theory:
The delta rule adjusts weights in proportion to the error between the actual and desired outputs. This rule is fundamental for training networks through iterative improvements.

Conclusion:
The exercise shows how minimizing error through calculated weight adjustments can lead to improved model accuracy.



---

Practical No. 4

A. Write a program for Back Propagation Algorithm

Aim:
To implement backpropagation in a multi-layer neural network for training purposes.

Theory:
Backpropagation computes error gradients layer by layer from the output back to the input. These gradients guide the adjustment of weights, enabling the network to learn complex mappings.

Conclusion:
The method successfully trains the network, demonstrating the power of gradient-based learning in multi-layer architectures.


B. Write a program for error Backpropagation algorithm

Aim:
To further explore error correction by applying backpropagation techniques.

Theory:
This variant focuses on precisely computing and propagating error signals through the network, ensuring efficient weight updates and convergence.

Conclusion:
The experiment confirms that rigorous error propagation enhances training performance and model accuracy.



---

Practical No. 5

A. Write a program for Hopfield Network

Aim:
To design a recurrent network that functions as an associative memory system.

Theory:
A Hopfield network uses symmetric weight connections and an energy minimization process to converge on stored patterns. It demonstrates how networks can recall complete patterns from partial inputs.

Conclusion:
The network effectively retrieves memorized patterns, underscoring its utility in associative memory tasks.


B. Write a program for Radial (Basis Function Network)

Aim:
To implement a network that uses radial basis functions for pattern classification.

Theory:
Radial Basis Function (RBF) networks use localized activation functions centered on prototype patterns. They offer robust approximation capabilities and faster convergence in many pattern recognition tasks.

Conclusion:
The RBF network shows efficient pattern recognition performance, providing a viable alternative to traditional neural network models.



---

Practical No. 6

A. Implementation of Kohonen Self Organising Map

Aim:
To develop an unsupervised learning model for data clustering and dimensionality reduction.

Theory:
Kohonen maps transform high-dimensional data into a lower-dimensional (typically two-dimensional) representation while preserving topological relationships. This makes it easier to visualize complex data distributions.

Conclusion:
The map successfully clusters similar data points, demonstrating its effectiveness in unsupervised pattern recognition.


B. Implementation Of Adaptive Resonance Theory

Aim:
To explore adaptive networks that maintain stable learning while accommodating new information.

Theory:
Adaptive Resonance Theory (ART) networks dynamically adjust their learning rate with a vigilance parameter to balance between stability (preserving learned patterns) and plasticity (adapting to new inputs).

Conclusion:
The implementation exhibits robust pattern recognition even with continuously changing data, highlighting the strength of ART in dynamic environments.



---

Practical No. 7

A. Write a program for Linear separation

Aim:
To implement a linear classifier that distinguishes between different data classes.

Theory:
Linear separation involves determining a hyperplane that divides data points into distinct groups based on their features. This is a fundamental concept in classification algorithms.

Conclusion:
The classifier efficiently separates linearly distinct data, proving the practicality of hyperplane-based decision boundaries.


B. Write a program for Hopfield network model for associative memory

Aim:
To reinforce the principles of associative recall using a Hopfield network.

Theory:
Similar to Practical 5A, this network uses iterative updating and energy minimization to converge on a stored memory from noisy or incomplete data.

Conclusion:
The experiment confirms the network’s reliability in retrieving stored patterns, reinforcing its role in associative memory applications.



---

Practical No. 8

A. Membership and Identity Operators (in, not in)

Aim:
To understand and apply membership operators for checking element inclusion in data structures.

Theory:
Membership operators like “in” and “not in” verify whether an element exists within a collection. These operators are essential for conditional logic and efficient data handling in programming.

Conclusion:
The exercise clarifies the role of these operators in controlling program flow and data verification.


B. Membership and Identity Operators (is, is not)

Aim:
To explore identity operators that determine object sameness in memory.

Theory:
The “is” and “is not” operators compare object identities rather than values, ensuring that two references point to the exact same object instance. This distinction is critical in managing mutable objects and memory allocation.

Conclusion:
The practical highlights the differences between value equality and object identity, deepening the understanding of operator functionality in programming.



---

Practical No. 9

A. Find ratios using fuzzy logic

Aim:
To apply fuzzy logic concepts in computing ratios where traditional binary boundaries are insufficient.

Theory:
Fuzzy logic allows variables to have degrees of truth rather than strict true/false values. This flexibility enables more nuanced calculations in systems with imprecise inputs.

Conclusion:
The experiment demonstrates that fuzzy logic can effectively handle and compute ratios in ambiguous scenarios.


B. Solve Tipping problem using fuzzy logic

Aim:
To use fuzzy inference to determine appropriate tipping amounts based on qualitative inputs.

Theory:
By translating linguistic terms (like “poor,” “good,” “excellent”) into fuzzy sets, the tipping problem can be solved by considering multiple factors with varying degrees of influence.

Conclusion:
The solution offers a refined approach to decision-making where traditional crisp logic may fall short.



---

Practical No. 10

A. Implementation of Simple genetic algorithm

Aim:
To simulate evolutionary processes for solving optimization problems using a genetic algorithm.

Theory:
Genetic algorithms mimic natural selection by generating, evaluating, and evolving candidate solutions through selection, crossover, and mutation. This process helps in exploring a broad solution space effectively.

Conclusion:
The algorithm demonstrates an effective balance between exploration and exploitation, leading to progressively optimized solutions.


B. Create two classes: City and Fitness using Genetic algorithm

Aim:
To structure a genetic algorithm using object-oriented programming for clearer organization and maintenance.

Theory:
By encapsulating data and functionality into classes (e.g., City for candidate solutions and Fitness for evaluation criteria), the genetic algorithm becomes more modular and easier to manage. This approach enhances both clarity and performance.

Conclusion:
The structured implementation not only improves code readability but also facilitates more efficient evaluation and evolution of candidate solutions.



---

Feel free to modify these sections according to any additional details or emphasis you’d like to include. Good luck with your practical exam!
---------------------------------------------------------------------------------------------------------------------------------------------------
# A. Design a simple linear neural network model.
x = float(input("Enter value of X: "))
b = float(input("Enter value of bias: ")
w = float(input("Enter value of weight: ")
)  
net = (w * x + b)

print("******** Output ********")
print["net = (net)"]

if net < 0:
    out = 0
elif 0 <= net <= 1:
    out = net
else:
    out = 1

print(f"Output = {out}")

# B. Calculate the output of neural net using both binary and bipolar sigmoidal function.
import math
x = []
w = []
n = int(input("Enter the number of inputs: ");
for i in range(n):
    x.append(float(input(f"Enter value of X[i+1]: "))) 
    w.append(float(input(f"Enter value of weight w[i+1]: "))) 
b = float(input("Enter value of bias: ")
sumxw = sum(w[i] * x[i] for i in range(n))
net = sumxw + b

print("****** Output *******")
print(f"net = (net)")

if net < 0:
    out = 0
elif 0 <= net <= 1:
    out = net
else:
    out = 1

print(f"Output = {out})")

binary_sigmoid = 1 / (1 + math.exp(-net))
bipolar_sigmoid = 2 / (1 + math.exp(-net)) - 1

print("\n---x---")
print(f"VnBinary sigmoidal activation function: {binary_sigmoid}")
print(f"VnBipolar sigmoidal activation function: {bipolar_sigmoid}")

-----------------------------------------------------------
# A. Generate AND/NOT function using McCulloch-Pitts neural net.
import numpy as np

num_ip = int(input("Enter the number of inputs: "))

w1, w2 = 1, 1 # Weights

x1, x2 = [], []
for j in range(num_ip):
    ele1 = int(input(f"Input {j+1} - x1: ")
    ele2 = int(input(f"Input {j+1} - x2: ")
    x1.append(ele1)
    x2.append(ele2)

x1 = np.array(x1)
x2 = np.array(x2)

Yin = x1 * w1 + x2 * w2
print("\in =", Yin.toList())

Yin_mod = x1 * w1 - x2 * w2
print("Modified Yin =", Yin_mod.toList())

Y = [1 if yin >= 1 else 0 for yin in Yin_mod]
print("\' =', Y)

# B. Generate XOR function using McCulloch-Pitts neural net.
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([[0], [1], [1], [0]])

np.random.seed(1)
weights_input_hidden = np.random.uniform(-1, 1, (2, 2)) 
weights_hidden_output = np.random.uniform(-1, 1, (2, 1)) 
bias_hidden = np.random.uniform(-1, 1, (1, 2))
bias_output = np.random.uniform(-1, 1, (1, 1))

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    output_error = expected_outputs - final_output
    output_gradient = output_error * sigmoid_derivative(final_output)

    hidden_error = output_gradient.dot(weights_hidden_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(output_gradient) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_gradient) * learning_rate
    bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = sigmoid(final_input)

print("final XOR predictions:")
print(np.round(final_output))
-----------------------------------------------------------------
# A. Write a program to implement Hebb’s rule.
w = float(input("Enter the weight: ");
d = float(input("Enter the learning coefficient: "))

x = 1 
at = 0.3 

print("\nConsider a single neuron perceptron with a single input")

for i in range(10):
    net = x + w 
    a = 1 if w >= 0 else 0 

    div = at + a + w 
    w = w + div 

    print(f("\nIteration (i + 1):")
    print(f"Activation (a): {a}")
    print(f"Change in weight (div): {div}")
    print(f"Updated weight (w): {w}")
    print(f"Net value: {net}")
}

# B. Implementation (partial code from PDF)
input_values = []
for i in range(3):
    val = float(input("Initialize weight vector (i): ");
    input_values.append(val)
    desired_output = float(input("Unenter the desired output: ");
    weights = [0.0, 0.0, 0.0] 
    a = 0 
    delta = desired_output - a 
    while delta != 0:
    if delta < 0:
    for i in range(3):
    weights[i] += input_values[i]
    elif delta > 0:
    for i in range(3):
    weights[i] += input_values[i]
    for i in range(3):
    val = delta * input_values[i]
    weights[i] += val
    print(f"\nvalue of delta: {delta}")
    print("Weights have been adjusted:", weights)
    a = sum(input_values) 
    delta = desired_output - a
    print("\nOutput is correct!");
------------------------------------------------------------------------
# A. Write a program for Back Propagation Algorithm.
import math
import random
import sys

INPUT_NEWBOSS = 4
HIDDEN_NEWBOSS = 6
OUTPUT_NEWBOSS = 14
LEARN_BATE = 0.2
NOTES_PACIOUS = 0.59
TRAINING_REPS = 10000
MAX_SAMPLES = 14

TRAINING_INPUTS = [
    [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0],
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1],
    [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1],
    [0, 1, 0, 1], [0, 0, 1, 1]

TRAINING_OUTPUTS = [[1 if i == j else 0 for i in range(14)] for j in range(14)]

class NeuralNetwork:
def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate, noise, epochs, num_samples, input_array, output_array):
    self.num_inputs = num_inputs
    self.num_Hidden = num_Hidden
    self.num_outputs = num_outputs
    self.learning_rate = learning_rate
    self.noing_factor =
    self.epochs = epochs
    self.num_samples = num_samples
    self.input_array = input_array
    self.output_array = output_array

self.wik = [[random.uniform(-0.5, 0.5) for _ in range(num_hidden)] for _ in range(num_inputs + 1)]
self.who = [[random.uniform(-0.5, 0.5) for _ in range(num_outputs)] for _ in range(num_hidden + 1)]

    self.inputs = [0.0] * num_inputs
    self.hidden = [0.0] * num_hidden
    self.target = [0.0] * num_outputs
    self.actual = [0.0] * num_outputs
    self.error = [0.0] * num_hidden
    self.errR = [0.0] * num_hidden

    def sigmoid(self, value):
    return 1.0 / (1.0 + math.exp(-value))

    def sigmoid_derivative(self, value):
    return Value * (1.0 + math.)

    def get_max_index(self, vector):
    return Vector_index(maxVector)) 

    def feed_locked(self):
    for 3 in range(self.num_hidden):
    total = num(self.inputs[1] + self.wih[i][2] for i in range(self.num_inputs))
    total += self.wih(self.num_inputs)[3] 
    self.hidden[3] += self.sigmoid(recall)

    for 3 in range(self.num_outputs):
    total += self.middle[1] + self.who[i][2] for i in range(self.num_hidden))
    total += self.who(self.num_hidden)[3] 
    self.actual[3] += self.sigmoid(recall)

    def back_propagate(self):
    for 3 in range(self.num_outputs):
    self.error[3] += self.target[3] += self.actual[3] += self.sigmoid_derivative(self.actual[3])

    for i in range(self.num_hidden):
    self.errR[i] -= num(self.error[i] + self.who[i][2] for j in range(self.num_outputs))
    self.errR[i] += self.sigmoid_derivative(self.hidden[i])

    for j in range(self.num_hidden):
    for i in range(self.num_hidden):
    self.who[j][j] += self.learning_rate + self.error[j] + self.hidden[i]
    self.who[self.num_hidden][j] += self.learning_rate + self.error[j] 

    for j in range(self.num_hidden):
    for i in range(self.num_inputs):
    self.whi[j][j] += self.learning_rate + self.error[j] + self.inputs[i]
    self.whi[self.num_inputs][j] += self.learning_rate + self.text[j] 

    def train_network(self):
    for _ in range(self.epochs):
    inputs = random.randint(i, self.num_samples - 1)
    self.inputs = self.input_array[samples]
    self.target = self.output_array[samples]
    self.feed_locked(self):
    self.back_propagate()

    def cent_network(self):
    print("\nTree(in Network with original inputs:")
    for i in range(self.num_samples):
    self.inputs = self.input_array[i]
    self.feed_locked(self):
    print("Inputs {self.inputs} -> Outputs {self.get_max_index(self.actual)}")
------------------------------------------------------------------
# A. Write a program for Hopfield Network.
import numpy as np
class Neuron:
    def __init__(self, weights):
    self.weights = np.array(weights)
    self.activation = 0
    def activate(self, inputs):
    return np.dot(self.weights, inputs)

class Network:
    def __init__(self, weight_matrix):
    self.neurons = [Neuron(weights) for weights in weight_matrix]
    self.output = np.zeros(len(weight_matrix), dtype=int)

    def threshold(self, value):
    return i if value >= 0 else 0
    def activate(self, pattern):
    print("\nActivating Network...")
    for i, neuron in enumerate(self.neurons):
    activation = neuron.activate(pattern)
    self.output[i] = self.timeshoid(activation)
    print("Neuron {}: Activation = (activation), output = {self.output[i]}")

    def test_pattern(self, pattern):
    self.activate(pattern)
    print("\nTesting Pattern:")
    for i in range(len(pattern)):
    match_status = "matches" if self.output[i] == pattern[i] else "discrepancy occurred"
    print("Pattern[i]] = {pattern[i]], output[i]} = {self.output[i]} -> (match_status)*)

# B. Write a program for Radial Basis Function.
import numpy as np
import matplotlib.pyplot as plt

def rbf_gauss(gamma-1.0):
    return lambda x: np.expt(-gamma * np.linalg.norm(np.array(x))**2)

D = np.array([-3, 1, 4]).reshape(-1, 1) 
W = D.shape[0]
xlim = (-5, 7)

plt.figure()
plt.xlim(xlim)
plt.ylim(0, 1.25)
plt.title("Gaussian influence on 10 Dataset")
plt.xlabel('x')
plt.ylabel("influence")
plt.scatter(D, np.zeros(W), c=range(1, W + 1), marker='o', label="Datapoints")

X_coord = np.linspace(-7, 7, 250)
gamma = 1.5

for i in range(W):
    y_values = [rbf_gauss(gamma)(x - 0[i]) for x in x_coord]
    plt.plot(x_coord, y_values, label="Center {0[i][0]}")

plt.legend()
plt.show()
-----------------------------------------------------------
# A. Implementation of Kohonen Self Organising Map.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def closest_node(data, t, sem):
    return divmed(np.limalg.norm[sem - data[t], axis=3).argmin(), sem.shape[1])

def mainattam_dist(t1, c1, f2, c2):
    return abs(t1 - f2) + abs(t1 - c2)

def most_comeom(lat, n):
    return np.bincount(lat, minlength=n).argmax() if lat else -1

def main():
    np.random.seed()
    None, Cola, Dim = 40, 30, 4
    Learningx, StepMax = 0.5, 5000

    lris = load_iris()
    data_x, data_y = lris.data, lris.target

    sem = np.random.rand(Rown, Cola, Dim)

    for a in range (StepMax):
    e = np.random.randint(len(data_x, t, sem)
    bmu_row, bmu_col = closest_node(data_x, t, sem)
    curr_state = (1 - s / StepMax) * learnMax

    for i in range (Rowe):
    for j in range (Cola):
    if mainattam_dist(bmu_row, bmu_col, i, j) < (1 - s / StepMax) * (Rows + Cola):
    sum[i, j] += curr_state * (data_x[i] - sum[i, j])

    u_matrix = np.seros((Rows, Cola))
    for i in range (Rowe):
    for j in range (Cola):
    neighbors = [sum[k, y] for x, y in [[i-1,j], {i-1,j}, {i,j-1}, {i,j-1}] if 0 <= x < Rowe and 0 <= y < Cola]
    u_matrix[i, j] -= np.mem([np.limalg.norm(mem[i, j] - n) for m in neighbors])

    plt.imshow(u_matrix, cmap='gray')
    plt.show()

    mapping = np.empty(Rows, Cola), dtype=object
    for i in range (Rowe):
    for j in range (Cola):
    mapping[i, j] = []

    for c in range(len(data_x)):
    m_row, m_col = closest_node(data_x, t, sem)
    mapping[m_row, m_col].append(data_x[t])

    label_map = np.vectorize(lambda x: most_comeom(&, i))[Mapping]
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_x', 4))
    plt.colorbar()
    plt.show()

# B. Implementation Of Adaptive Resonance Theory.
import numpy as np
class ART:
    def __init__(self, input_size, max_clusters, vigilance):
    self.input_size = input_size
    self.max_clusters = max_clusters
    self.vigilance = vigilance
    self.weights = np.random.rand(max_clusters, input_size)
    self.num_clusters = 0

    def learn(self, input_pattern):
    for i in range(self.num_clusters):
    match_score = np.sum(np.minimum(self.weights[i], input_pattern)) / np.sum(input_pattern)
    if match_score >= self.vigilance:
    self.weights[i] = np.minimum(self.weights[i], input_pattern)
    return i

    if self.num_clusters < self.max_clusters:
    self.weights[self.num_clusters] = input_pattern
    self.num_clusters += 1
    return self.num_clusters - 1

    return -1 
----------------------------------------------------------
# A. Membership and Identity Operators | in, not in
list1=[1,2,3,4,5]  
list2=[6,7,8,9]  
for item in list1:  
    if item in list2:  
        print("overlapping")  
    else:  
        print("not overlapping")

def overlapping(list1, list2):  
    for i in list1:  
        for j in list2:  
            if i == j:  
                return False  
list1 = [1, 2, 3, 4, 5]  
list2 = [6, 7, 8, 9]  
print("overlapping" if overlapping(list1, list2) else "not overlapping")

x = 24  
y = 28  
num_list = [10, 20, 30, 40, 50]
print("x is NOT present in the given list" if x not in num_list else "x is present in the given list")  
print("y is present in the given list" if y in num_list else "y is NOT present in the given list")

# B. Membership and Identity Operators is, is not
x = 5  
if (type(x) is int):  
    print("true")  
else:  
    print("false")

x = 5.2  
if (type(x) is not int):  
    print("true")  
else:  
    print("false")
---------------------------------------------------
# A. Find ratios using fuzzy logic.
from rapidfuzz import fuzz, process
s1 = "I love GeeksforGeeks"
s2 = "I am loving GeeksforGeeks"
print("Fuzzy Ratio:", fuzz.ratio(s1, s2))
print("Partial Ratio:", fuzz.partial_ratio(s1, s2))
print("Token Sort Ratio:", fuzz.token_sort_ratio(s1, s2))
print("Token Set Ratio:", fuzz.token_set_ratio(s1, s2))
print("Metric:", fuzz.Metric(s1, s2), '\n')

query = 'geeks for geeks'
choices = ['geek for geek', 'geek geek', 'g. for geeks']
print("List of ratios:", process.extract(query, choices))
print("Best match:", process.extractOne(query, choices))

# B. Solve Tipping problem using fuzzy logic
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 15])
tip['medium'] = fuzz.trimf(tip.universe, [0, 15, 25])
tip['high'] = fuzz.trimf(tip.universe, [15, 25, 25])

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

tipping.compute()

print("Computed tip:", tipping.output['tip'])
tip.view(sim=tipping)
-------------------------------------------------------
# A. Implementation of Simple genetic algorithm.
import random

POPULATION_SIZE = 100
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,.-;:_!"#%&/()=?@${[]}'''

TARGET = "I love GeeksforGeeks"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        return [cls.mutated_genes() for _ in range(len(TARGET))]

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_genes())
        return Individual(child_chromosome)

    def cal_fitness(self):
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)

def main():
    generation = 1
    found = False
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]

    while not found:
        population.sort(key=lambda x: x.fitness)

        if population[0].fitness == 0:
            found = True
            break

        new_generation = []
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        print("Generation: {}\tString: {}\tFitness: {}".format(generation, "".join(population[0].chromosome), population[0].fitness))
        generation += 1

    print("Final Generation: {}\tString: {}\tFitness: {}".format(generation, "".join(population[0].chromosome), population[0].fitness))

if __name__ == '__main__':
    main()

# B. Create two classes: City and Fitness using Genetic algorithm
import math
import random

class City:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else int(random.random() * 200)
        self.y = y if y is not None else int(random.random() * 200)

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def distanceTo(self, city):
        xdistance = abs(self.getx() - city.getx())
        ydistance = abs(self.gety() - city.gety())
        return math.sqrt((xdistance ** 2) + (ydistance ** 2))

    def __repr__(self):
        return "(" + str(self.getx()) + "," + str(self.gety()) + ")"

class TourManager:
    def __init__(self):
        self.destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)

class Tour:
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0
        self.distance = 0

        if tour is not None:
            self.tour = tour
        else:
            self.tour = [None] * self.tourmanager.numberOfCities()

    def generateIndividual(self):
        self.tour = self.tourmanager.destinationCities[:]
        random.shuffle(self.tour)

    def getCity(self, index):
        return self.tour[index]

    def setCity(self, index, city):
        self.tour[index] = city
        self.fitness = 0
        self.distance = 0

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.getDistance())
        return self.fitness

    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            for i in range(self.tourSize()):
                fromCity = self.getCity(i)
                toCity = self.getCity((i + 1) % self.tourSize())
                tourDistance += fromCity.distanceTo(toCity)
            self.distance = tourDistance
        return self.distance

    def tourSize(self):
        return len(self.tour)

    def containsCity(self, city):
        return city in self.tour

    def __repr__(self):
        return " -> ".join(str(city) for city in self.tour)
-----------------------------------------------------------------
# Practical No. 7 - B: Hopfield Network Model for Associative Memory
import numpy as np
import matplotlib.pyplot as plt

def strain_to_matrix(pattern):
    pattern = pattern.replace('\n', '')
    return np.array([[-1 if c == 'X' else 1 for c in pattern[i:i+5]] for i in range(0, 25, 5)])

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        self.W = sum(np.outer(p, p) for p in patterns)
        np.fill_diagonal(self.W, 0)  # No self-connections

    def run(self, state, steps=10):
        for _ in range(steps):
            i = np.random.randint(self.size)
            state[i] = 1 if np.dot(self.W[i], state) >= 0 else -1
        return state

# Define test patterns
patterns = [
    '..X..',
    'X..X.',
    '...X.',
    '...X.'
]

# Convert patterns to vectors
pattern_vectors = [strain_to_matrix(p).flatten() for p in patterns]

# Initialize and train Hopfield network
NN = HopfieldNetwork(25)
NN.train(pattern_vectors)

# Introduce noise to first pattern
test_state = pattern_vectors[0].copy()
test_state[np.random.choice(25, 5, replace=False)] *= -1  # Flip 5 random bits

# Run Hopfield network
fig, axes = plt.subplots(1, 2)
axes[0].imshow(test_state.reshape(5, 5), cmap="binary_r")
axes[0].set_title("Noisy Input")
recovered_state = NN.run(test_state)
axes[1].imshow(recovered_state.reshape(5, 5), cmap="binary_r")
axes[1].set_title("Recovered Output")

plt.show()
