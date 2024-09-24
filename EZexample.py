import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 微分的值
def sigmoid_derivative(x):
    """
    計算 sigmoid 函數的導數。
    
    數學觀念：
    1. sigmoid 函數: f(x) = 1 / (1 + e^(-x))
    2. sigmoid 導數: f'(x) = f(x) * (1 - f(x))
    
    推導過程：
    1. 設 y = f(x) = 1 / (1 + e^(-x))
    2. dy/dx = (1 + e^(-x))^(-2) * e^(-x)
             = y * y * e^(-x)
             = y * y * (1 + e^(-x) - 1)
             = y * (y * (1 + e^(-x)) - y)
             = y * (1 - y)
    
    因此，sigmoid 的導數可以表示為輸出值乘以 (1 - 輸出值)。
    這個函數假設輸入 x 已經是 sigmoid 函數的輸出。
    """
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化權重和偏置
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) # 隨機產生input to hidden的weight配置
        self.bias_hidden = np.zeros((1, self.hidden_size)) # 設定hidden的bias為0
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) # 隨機產生hidden to output的配置 
        self.bias_output = np.zeros((1, self.output_size)) # 設定output的bias為0
    
    def forward(self, X):
        # 前向傳播
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output 

    # epoch: 訓練次數！ 
    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            # 前向傳播 
            self.forward(X) 
            
            # 反向傳播
            output_error = y - self.output
            output_delta = output_error * sigmoid_derivative(self.output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            
            hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

            
            # 更新隱藏層到輸出層的權重
            self.weights_hidden_output += np.dot(self.hidden.T, output_delta) * learning_rate
            # 更新輸出層的偏置
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            # 更新輸入層到隱藏層的權重
            self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
            # 更新隱藏層的偏置
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# 隱藏層想設定幾個就設定幾個
nn = NeuralNetwork(2, 10, 1)
# 這次的訓練代表的是XOR的輸入, 以變數X做輸入
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y 是預期結果
y = np.array([[0], [1], [1], [0]])

# 可以refine的參數！
nn.train(X, y, learning_rate=0.1, epochs=100000)

# 測試網絡
for i in range(4):
    print(f"輸入: {X[i]}, 預測輸出: {nn.forward(X[i])}")
