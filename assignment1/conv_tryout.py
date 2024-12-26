from EDF_Percpetron import *
# given
w = np.random.rand(3, 3, 3, 16)
b = np.random.rand(16)
activation = np.random.rand(16, 11, 11, 3)
padding = 'valid'

w_n = Parameter(w)
b_n = Parameter(b)
inp = Input()
inp.value = activation
# when
node = Conv(param=[b_n, w_n], var=inp)
node.forward()
result = node.value

assert result.shape == (16, 11, 11, 16)
expected_val = np.sum(w[:, :, :, 0] * activation[0, 0:3, 0:3, :]) + b[0]
assert abs(expected_val - result[0, 1, 1, 0]) < 1e-8