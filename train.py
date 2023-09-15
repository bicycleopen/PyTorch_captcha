import torch
from torch import nn
from loader import get_loader
from generate import CHAR_NUMBER, SEED


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class NeuralNetWork(nn.Module):
    def __init__(self):
        super(NeuralNetWork, self).__init__()
        self.layer1 = nn.Sequential(
                    # 输入通道为1，   输出为64，        卷积核为3，     扩充边缘为1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=15360, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=CHAR_NUMBER*len(SEED))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"损失值: {loss:>7f}")


def main():
    model = NeuralNetWork().to(device)
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_loader("./train_captcha")
    
    epoch = 2
    for t in range(epoch):
        print(f"训练周期 {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("\n")

    torch.save(model.state_dict(), "model.pth")
    print("训练完成，模型已保存")


if __name__ == "__main__":
    main()