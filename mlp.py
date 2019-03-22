import numpy as np
import imageio
import math
import functools

#Classe que representa o multilayer perceptron
class MLP():
	def __init__(self, input_length, hidden_length, output_length):
		self.input_length = input_length
		self.hidden_length = hidden_length
		self.output_length = output_length
		self.hidden_mat = np.random.uniform(-0.5, 0.5, (hidden_length, input_length+1))
		self.output_mat = np.random.uniform(-0.5, 0.5, (output_length, hidden_length+1))

	#Funcao de ativacao (sigmoide)
	def activ(self, net):
		return (1/(1+math.exp(net)))

	def deriv_activ(self, net):
		return net*(1-net)

	#Faz inferencia
	def forward(self, input_vect):
		if(input_vect.shape[0] != self.input_length):
			message = 'Tamanho incorreto de entrada. Recebido: {} || Esperado: {}'.format(input_vect.shape[0], self.input_length)
			raise Exception(message)
		biased_input = np.zeros((input_vect.shape[0]+1))
		biased_input[0:input_vect.shape[0]] = input_vect[:]
		biased_input[input_vect.shape[0]] = 1

		hidden_net = np.dot(self.hidden_mat, biased_input)
		hidden_fnet = [self.activ(x) for x in hidden_net]

		biased_hidden_activ = np.zeros((self.hidden_length+1))
		biased_hidden_activ[0:self.hidden_length] = hidden_fnet[:]
		biased_hidden_activ[self.hidden_length] = 1
		out_net = np.dot(self.output_mat, biased_hidden_activ)
		out_fnet = [self.activ(x) for x in out_net]

		return hidden_net, hidden_fnet, out_net, out_fnet



	#Faz backpropagation
	def fit(self, input_samples, target_labels, learning_rate, threshold):
		squared_error = 2*threshold
		while(squared_error > threshold):
			squared_error = 0
			for i in range(0, input_samples.shape[0]):
				input_sample = input_samples[i]
				input_sample_with_bias = np.zeros(input_sample.shape)
				input_sample_with_bias[0:input_sample.shape[0]] = input_sample[:]
				input_sample_with_bias[input_sample.shape[0]]

				hidden_net, hidden_fnet, out_net, out_fnet = self.forward(input_samples[i])
				target_label = target_labels[i]
				error_array = (target_label - out_fnet)
				squared_error = squared_error + np.sum(error_array**2)

				delta_o = error * deriv_activ(out_fnet)
				output_weights = self.output_mat[:,0:hidden_length]
				delta_h = deriv_activ(hidden_fnet) * np.dot(delta_o, output_weights)

				self.output_mat = self.output_mat + learning_rate*np.dot(delta_o, hidden_fnet)
				self.hidden_mat = self.hidden_mat + learning_rate*np.dot(delta_h.T, input_sample_with_bias)
			squared_error = squared_error/input_samples.shape[0]
			print('Erro medio quadratico', squared_error)
		return None

def squared_error(target, output):
	squared_differences = [(target[x] - output[x])**2 for x in range(len(target))]
	return functools.reduce(lambda x,y: x+y, squared_differences)

def main():
	mlp = MLP(*(2, 2, 2))
	out = mlp.forward(np.random.uniform(-1., 1., (2)))
	print('output', out)
	vec1 = [1, 0, 0, 1]
	vec2 = [1, 0, 0, -1]

if __name__ == '__main__':
	main()