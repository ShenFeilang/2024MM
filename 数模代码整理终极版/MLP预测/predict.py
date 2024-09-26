import torch
from train_mlp import MLP, train_db
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = MLP(len(train_db.datas[1]), 1)  # 初始化模型
model.load_state_dict(torch.load('best.mdl'))
model.eval()

data = pd.read_csv('dataset/except_pdu_dur.csv')

features = data.iloc[:, 1:-1].values  # 所有列，除了最后一列

features = StandardScaler().fit_transform(features).tolist()  # 对数据进行预处理

# 将数据转换为PyTorch张量
features_tensor = torch.tensor(features, dtype=torch.float32)

# 进行预测
with torch.no_grad():
    predictions = model(features_tensor)

# 打印预测结果
pd.DataFrame(predictions).to_csv('target.csv')
