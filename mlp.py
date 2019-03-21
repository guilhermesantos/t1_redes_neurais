import numpy as np
import imageio
import math

class MLP():
	def __init__(self, input_length, hidden_length, output_length):
		self.input_length = input_length
		self.hidden_length = hidden_length
		self.output_length = output_length
		self.hidden_mat = np.random.uniform(-0.5, 0.5, (hidden_length, input_length+1))
		self.output_mat = np.random.uniform(-0.5, 0.5, (output_length, hidden_length+1))

	def activ(self, net):
		return (1/(1+math.exp(net)))

	def forward(self, input_vect):
		if(input_vect.shape[0] != self.input_length):
			message = 'Tamanho incorreto de entrada. Recebido: {} || Esperado: {}'.format(input_vect.shape[0], self.input_length)
			raise Exception(message)
		biased_input = np.zeros((input_vect.shape[0]+1))
		biased_input[0:input_vect.shape[0]] = input_vect[:]
		biased_input[input_vect.shape[0]] = 1

		hidden_net = np.dot(self.hidden_mat, biased_input)
		hidden_activ = [self.activ(x) for x in hidden_net]

		biased_hidden_activ = np.zeros((self.hidden_length+1))
		biased_hidden_activ[0:self.hidden_length] = hidden_activ[:]
		biased_hidden_activ[self.hidden_length] = 1
		out_net = np.dot(self.output_mat, biased_hidden_activ)
		out_activ = [self.activ(x) for x in out_net]

		return out_activ

	def fit(self, input_samples, expected_labels):
		return None

def main():
	mlp = MLP(*(2, 2, 2))
	out = mlp.forward(np.random.uniform(-1., 1., (2)))
	print('output', out)
if __name__ == '__main__':
	main()