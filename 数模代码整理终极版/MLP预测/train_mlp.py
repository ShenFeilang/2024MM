import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from MyDataSet import MyDataSet
from MLP import MLP
from tqdm import tqdm
import matplotlib.pyplot as plt


batchsz = 16
lr = 1e-3  # 学习率
epoches = 10
torch.manual_seed(1234)
file_path = "dataset/except_pdu_dur.csv"

# 读取数据
train_db = MyDataSet(file_path, mode='train')
val_db = MyDataSet(file_path, mode='val')
test_db = MyDataSet(file_path, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)


def main():
    model = MLP(len(train_db.datas[1]), 1)  # 初始化模型

    optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置Adam优化器

    # criterion = nn.CrossEntropyLoss()    # 设置损失函数
    # optimizer = optim.SGD(model.parameters(), lr=lr)  # 设置Adam优化器

    criterion = nn.MSELoss()  # 设置损失函数
    # criterion_1=nn.L1Loss()     # 绝对值损失桉树

    best_epoch, best_loss = 0, 1000
    train_losses = []
    val_losses = []

    for epoch in range(epoches):
        epoch_loss = 0
        for step, (x, y) in enumerate(tqdm(train_loader)):
            logits = model(x)
            loss = criterion(logits, y.reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        if epoch % 5 == 0:
            correct_loss = 0
            total = len(val_loader.dataset)

            for x, y in val_loader:
                with torch.no_grad():
                    logits = model(x)
                    loss = criterion(logits, y.reshape(-1, 1))
                correct_loss += loss.item()

            val_loss = correct_loss / total
            val_losses.append(val_loss)
            print("epoch:[{}/{}]. val_loss:{}.".format(epoch, epoches, val_loss))

            # print("train_acc", train_acc)
            if val_loss < best_loss:
                best_epoch = epoch
                best_loss = val_loss

                torch.save(model.state_dict(), 'best.mdl')



    print('best loss:{}. best epoch:{}.'.format(best_loss, best_epoch))
    model.load_state_dict(torch.load('best.mdl'))

    # Plotting the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(epoches), train_losses, label='Train Loss')
    plt.plot(range(0, epoches, 5), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()


    print("loaded from ckpt!")

    correct_loss = 0
    total = len(test_loader.dataset)
    for x, y in test_loader:
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y.reshape(-1, 1))
        correct_loss += loss
    test_acc = correct_loss / total
    print("test_acc:{}".format(test_acc))


if __name__ == '__main__':
    main()
