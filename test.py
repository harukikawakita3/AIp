# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms

# # データの前処理を定義
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # MNISTデータセットを読み込む
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# # バッチサイズとデータローダーの準備
# batch_size = 64
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# # ニューラルネットワークの定義
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)

# # ニューラルネットワークの初期化
# model = Net()

# # 損失関数と最適化アルゴリズムの定義
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# # トレーニングループ
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# # テストループ
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# # トレーニングとテストの実行
# for epoch in range(1, 11):
#     train(epoch)
#     test()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms

# # GPUが利用可能かどうかを確認します
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())

# # データセットをダウンロードします
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=64, shuffle=True)

# # CNNモデルを定義します
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.dropout = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
#         x = nn.functional.relu(nn.functional.max_pool2d(self.dropout(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)

# # モデルをGPUに移動します
# model = Net().to(device)

# # 損失関数を定義します
# criterion = nn.CrossEntropyLoss()

# # オプティマイザを定義します
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# # モデルをトレーニングします
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# # 10エポック分のトレーニングを実行します
# for epoch in range(1, 11):
#     train(epoch)


import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
