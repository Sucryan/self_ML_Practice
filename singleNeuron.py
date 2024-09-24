import numpy as np
'''
這份code基本上是weight亂數, bias也亂數，理論上假設是真實情況，我應該要用梯度下降法或者甚至是手動設置他的參數，讓sigmoid跑出來的數值盡可能接近1，才是好的做法！
'''
class Neuron:
	def __init__(self, num_inputs):
		# randn 代表給他常態分佈的亂數, rand代表給他隨機亂數
		self.weight = np.random.randn(num_inputs) # 隨機給權重 
		self.bias = np.random.randn() # 隨機給bias
	
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x)) # 定義一個activation function, 這邊用sigmoid

	def forward(self, inputs):
		total = np.dot(self.weight, inputs) + self.bias # np.dot() 矩陣相乘, 所以這個是把input跟隨機的加權乘起來！, 並且加上bias
		return self.sigmoid(total)

tryNeuron = Neuron(2) # 創建一個接收兩個輸入的神經元
input_data = np.array([2, 3])
output = tryNeuron.forward(input_data)
print(f"神經元輸出: {output}")  
print(f"weight:{tryNeuron.weight}, bias:{tryNeuron.bias}")	
