import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

# 入力データを定義し、GPUに移行する
inputs = torch.randn(1, 10).to(device)

# モデルに入力を与えて計算を実行する。結果もGPU上に生成される。
outputs = model(inputs)

print(inputs)