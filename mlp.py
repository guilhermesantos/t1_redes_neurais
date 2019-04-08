import numpy as np
import imageio
import math
import functools

#Classe que representa o multilayer perceptron
class MLP():
	#Construtor. Recebe o tamanho das cadamas de entrada, oculta e de saidas
	def __init__(self, input_length, hidden_length, output_length):
		self.input_length = input_length
		self.hidden_length = hidden_length
		self.output_length = output_length

		#Inicializa os pesos da camada oculta aleatoriamente, representando-os na forma de matriz
		#Os pesos e vies de cada neuronio sao dispostos em linhas
		#Em input_length+1, o +1 serve para representar o vies
		self.hidden_mat = np.random.uniform(-0.5, 0.5, (hidden_length, input_length+1))

		#Inicializa os pesos da camada de saida aleatoriamente, representado-os na forma de matriz
		#Os pesos e vies de cada neuronio sao dispostos em linhas
		#Em hidden_length+1, o +1 serve para representar o vies
		self.output_layer = np.random.uniform(-0.5, 0.5, (output_length, hidden_length+1))

	#Funcao de ativacao (sigmoide)
	def activ(self, net):
		return (1/(1+math.exp(net)))

	#Derivada da funcao de ativacao (sigmoide)
	def deriv_activ(self, fnet):
		one_vector = np.ones(fnet.shape)
		return fnet*(one_vector-fnet)

	def forward(self, input_vect):
		return self.forward_training(input_vect)[3]

	#Faz forward propagation (calcula a predicao da rede)
	def forward_training(self, input_vect):
		#Checa se o tamanho da entrada corresponde ao que eh esperado pela rede
		if(input_vect.shape[0] != self.input_length):
			message = 'Tamanho incorreto de entrada. Recebido: {} || Esperado: {}'.format(input_vect.shape[0], self.input_length)
			raise Exception(message)

		#Adiciona um componente "1" ao vetor de entrada para permitir calculo do bias
		#na camada oculta
		biased_input = np.zeros((input_vect.shape[0]+1))
		biased_input[0:input_vect.shape[0]] = input_vect[:]
		biased_input[input_vect.shape[0]] = 1

		#Calcula a transformacao da entrada pela camada oculta usando produto de matriz por vetor 
		#Wh x A = net, sendo Wh a matriz de pesos da camada oculta e A o vetor de entrada 
		hidden_net = np.dot(self.hidden_mat, biased_input)
		#Aplica a funcao de ativacao sobre a transformacao feita pela camada oculta
		hidden_fnet = np.array([self.activ(x) for x in hidden_net])

		#Adiciona um componente "1" ao vetor produzido pela camada oculta para permitir calculo do bias
		#na camada de saida
		biased_hidden_activ = np.zeros((self.hidden_length+1))
		biased_hidden_activ[0:self.hidden_length] = hidden_fnet[:]
		biased_hidden_activ[self.hidden_length] = 1
		
		#Calcula a transformacao feita pela camada de saida usando produto de matriz por vetor
		#Wo x H = net, sendo Wo a matriz de pesos da camada de saida e H o vetor produzido pela ativacao
		#da camada oculta
		out_net = np.dot(self.output_layer, biased_hidden_activ)
		#Aplica a funcao de ativacao nos valores produzidos pela transformacao da camada de saida
		out_fnet = np.array([self.activ(x) for x in out_net])

		#Retorna net e f(net) da camada oculta e da camada de saida
		return hidden_net, hidden_fnet, out_net, out_fnet

	#Faz backpropagation
	def fit(self, input_samples, target_labels, learning_rate, threshold):
		print('backpropagating')
		#Erro quadratico medio eh inicializado com um valor arbitrario (maior que o threshold de parada)
		#p/ comecar o treinamento
		mean_squared_error = 2*threshold

		#Inicializa o numero de epocas ja computadas
		epochs = 0

		#Enquanto não chega no erro quadratico medio desejado ou atingir 5000 epocas, continua treinando
		while(mean_squared_error > threshold or epochs < 5000):

			#Erro quadratico medio da epoca eh inicializado com 0
			mean_squared_error = 0
			
			#Passa por todos os exemplos do dataset
			for i in range(0, input_samples.shape[0]):
				#Pega o exemplo da iteracao atual
				input_sample = input_samples[i]
				#Pega o label esperado para o exemplo da iteracao atual
				target_label = target_labels[i]

				#Pega net e f(net) da camada oculta e da camada de saida
				hidden_net, hidden_fnet, out_net, out_fnet = self.forward_training(input_samples[i])
				
				#Cria um vetor com o erro de cada neuronio da camada de saida
				error_array = (target_label - out_fnet)

				#Atualiza os pesos da camada de saida com a regra delta generalizada
				#deltak = (Ypk-Ok)*Ok(1-Ok), sendo p a amostra atual do conjunto de treinamento,
				#e k um neuronio da camada de saida. Ypk eh a saida esperada do neuronio pelo exemplo do dataset,
				#Ok eh a saida de fato produzida pelo neuronio 
				delta_output_layer = error_array * self.deriv_activ(out_fnet)

				output_weights = self.output_layer[:,0:self.hidden_length]
				delta_hidden_layer = self.deriv_activ(hidden_fnet) * np.dot(delta_output_layer, output_weights)

				print('output layer', self.output_layer)
				print('delta', delta_output_layer)
				print('hidden fnet', hidden_fnet)
				#Wkj(t+1) = wkj(t) + eta*delta*Ipj
				for neuron in range(0, self.output_length):
					for weight in range(0, self.output_layer.shape[1]):
						print('updating weight index ', neuron, ' ', weight)
						print('current weight', self.output_layer[neuron, weight])
						print('current hidden fnet', hidden_fnet[weight])
						self.output_layer[neuron, weight] = self.output_layer[neuron, weight] + \
						learning_rate * delta_output_layer * hidden_fnet[weight]
				#self.output_layer = self.output_layer + learning_rate*np.dot(delta_output_layer, hidden_fnet)
				#self.output_layer = self.output_layer + learning_rate*(delta_output_layer @ hidden_fnet)

				#Atualiza os pesos da camada oculta com a regra delta generalizada
				#Pega os pesos dos neuronios da camada de saida (bias nao entra)
				input_sample_with_bias = np.zeros(input_sample.shape[0]+1)
				input_sample_with_bias[0:input_sample.shape[0]] = input_sample[:]
				input_sample_with_bias[input_sample.shape[0]] = 1
				self.hidden_mat = self.hidden_mat + learning_rate*np.dot(delta_hidden_layer.T, input_sample_with_bias)
				#self.hidden_mat = self.hidden_mat + learning_rate*(delta_hidden_layer.T @ input_sample_with_bias)

				#O erro da saída de cada neuronio é elevado ao quadrado e somado ao erro total da epoca
				#para calculo do erro quadratico medio ao final
				mean_squared_error = mean_squared_error + np.sum(error_array**2)				
			
			#Divide o erro quadratico total pelo numero de exemplos para obter o erro quadratico medio
			mean_squared_error = mean_squared_error/input_samples.shape[0]
			print('Erro medio quadratico', mean_squared_error)
			epochs = epochs + 1
		return None

def squared_error(target, output):
	squared_differences = [(target[x] - output[x])**2 for x in range(len(target))]
	return functools.reduce(lambda x,y: x+y, squared_differences)

def main():
	mlp = MLP(*(2, 2, 1))
	out = mlp.forward(np.array([0,1]))
	print('output', out)

	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	target = np.array([0, 1, 1, 0])
	mlp.fit(x, target, 5e-1, 10e-3)
if __name__ == '__main__':
	main()