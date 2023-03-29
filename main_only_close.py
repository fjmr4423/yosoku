import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas_datareader.data as pdr
import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

yf.pdr_override()

# データ取得期間の設定
dt_now = datetime.datetime.now()
end_date = dt_now
start_date = end_date - datetime.timedelta(weeks=1000)

# 銘柄コードの指定 (ここでは例としてAppleの株価データを取得します)
stock_code = 'JPY=X'

# データの取得
stock_data = pdr.get_data_yahoo(stock_code, start_date, end_date)
closing_prices = stock_data['Close'].values.astype('float32')
closing_prices = closing_prices.reshape(-1, 1)

# データの正規化
scaler = MinMaxScaler(feature_range=(-1, 1))
closing_prices_normalized = scaler.fit_transform(closing_prices)

# データを学習用とテスト用に分割
train_size = int(len(closing_prices_normalized) * 0.8)
train_data = closing_prices_normalized[:train_size]
test_data = closing_prices_normalized[train_size:]

# シーケンスデータを作成
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length - 1):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 20
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# PyTorch Tensorsに変換
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

# LSTMモデル定義
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # LSTMへの入力
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)

        # 結果の出力
        out = self.fc(h_out)
        return out

# ハイパーパラメータの設定
input_size = 1
hidden_size = 128
num_layers = 1
num_classes = 1
num_epochs = 100
learning_rate = 0.01

# モデルと損失関数、最適化手法の設定
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# 学習
for epoch in range(num_epochs):
    lstm.train()
    outputs = lstm(x_train)
    optimizer.zero_grad()

    # 損失計算とバックプロパゲーション
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# テストデータで予測
lstm.eval()
predicted = lstm(x_test)

# 正規化解除
predicted = scaler.inverse_transform(predicted.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# 本日の終値予測
today_price_data = closing_prices_normalized[-seq_length:]
today_price_data = torch.tensor(today_price_data).unsqueeze(0)
predicted_today = lstm(today_price_data)
predicted_today = scaler.inverse_transform(predicted_today.detach().numpy())

print(f"予測された本日の終値: {predicted_today[0][0]} ({end_date.strftime('%Y-%m-%d')})")

