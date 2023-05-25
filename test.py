# import torch
# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.linear = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.linear(x)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = MyModel().to(device)

# # 入力データを定義し、GPUに移行する
# inputs = torch.randn(1, 10).to(device)

# # モデルに入力を与えて計算を実行する。結果もGPU上に生成される。
# outputs = model(inputs)

# print(inputs)

# 手書き数字CNN

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = nn.functional.log_softmax(x, dim=1)
#         return output

# net = Net()
# net.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0

# print('Finished Training')

# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# print(device)

import youtube_dl

video_id = 'Vt_nc9pZnqs'

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '{}.mp3'.format(video_id),
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192'
    }]
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['http://www.youtube.com/watch?v=' + video_id])