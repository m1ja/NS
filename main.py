import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


tr_inputs = np.array([[0, 0, 1],
                      [1, 1, 1],
                      [1, 0, 1],
                      [0, 1, 1]])

tr_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

sc_weights = 2 * np.random.random((3, 1)) - 1

print('Случайные веса:')
print(sc_weights)

for i in range(20000):
    inp_layer = tr_inputs
    outputs = sigmoid(np.dot(inp_layer, sc_weights))

    err = tr_outputs - outputs
    ads = np.dot(inp_layer.T, err * (outputs * (1 - outputs)))

    sc_weights += ads

print('Веса после обучения :')
print(sc_weights)

print('Результат:')
print(outputs)


# Test
new_inputs = np.array([1, 1, 0])
outputs = sigmoid(np.dot(new_inputs, sc_weights))

print('Результаты после теста :')
print(outputs)
