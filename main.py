import streamlit as st
#------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas_datareader.data as pdr
import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import matplotlib.pyplot as plt
#------------------------------------------------
#サイドバー
#------------------------------------------------
st.sidebar.title('為替予測アプリ')

brand = st.sidebar.selectbox(
    '1.銘柄を入力してください。',
    ['USD/JPY','XAU/USD']
)
if brand == 'USD/JPY':
    brand = 'JPY=X'
elif brand == 'XAU/USD':
    brand = 'GC=F'

#------------------------------------------------
#予測ボタン
#------------------------------------------------
st.sidebar.write('2.予測ボタンを押下してください')
if st.sidebar.button("予測ボタン"):
    yf.pdr_override()

    # データ取得期間の設定
    dt_now = datetime.datetime.now()
    end_date = dt_now
    start_date = end_date - datetime.timedelta(weeks=1000)

    # 銘柄コードの指定 (ここでは例としてAppleの株価データを取得します)
    stock_code = brand

    # データの取得
    stock_data = pdr.get_data_yahoo(brand, start_date, end_date)

    # EMA(期間15日)と終値の乖離率を計算
    stock_data['EMA_15'] = stock_data['Close'].ewm(span=15).mean()
    stock_data['Disparity'] = (stock_data['Close'] - stock_data['EMA_15']) / stock_data['EMA_15']

    # 終値と乖離率をまとめる
    price_and_disparity = stock_data[['Close', 'Disparity']].values.astype('float32')

    # データの正規化
    scaler_price = MinMaxScaler(feature_range=(-1, 1))
    scaler_disparity = MinMaxScaler(feature_range=(-1, 1))

    closing_prices_normalized = scaler_price.fit_transform(price_and_disparity[:, 0].reshape(-1, 1))
    disparity_normalized = scaler_disparity.fit_transform(price_and_disparity[:, 1].reshape(-1, 1))

    # 終値と乖離率を組み合わせる
    data_normalized = np.hstack((closing_prices_normalized, disparity_normalized))

    # データを学習用とテスト用に分割
    train_size = int(len(data_normalized) * 0.8)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]

    # シーケンスデータを作成
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length - 1):
            x.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    seq_length = 30
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
    input_size = 2 # 入力サイズを2に変更
    hidden_size = 128
    num_layers = 1
    num_classes = 2 # 出力サイズを2に変更
    num_epochs = 100
    learning_rate = 0.01

    # モデルと損失関数、最適化手法の設定
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    st.sidebar.write('進捗状況(x/100)')

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
            st.sidebar.write(f'Epoch: {epoch + 1}, Loss: {loss.item()}', style={"font-size": "5px"})

    # テストデータで予測
    lstm.eval()
    predicted = lstm(x_test)

    # 正規化解除
    predicted = scaler_price.inverse_transform(predicted.detach().numpy()[:, 0].reshape(-1, 1))
    y_test = scaler_price.inverse_transform(y_test.detach().numpy()[:, 0].reshape(-1, 1))

    # 本日の終値予測
    today_price_data = data_normalized[-seq_length:]
    today_price_data = torch.tensor(today_price_data).unsqueeze(0)
    predicted_today = lstm(today_price_data)

    # 本日の終値予測の正規化解除
    predicted_today = scaler_price.inverse_transform(predicted_today.detach().numpy()[:, 0].reshape(-1, 1))
 
#------------------------------------------------
#現在価格の出力用
#------------------------------------------------

    # データ取得期間の設定
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)
    yosoku_date =end_date + datetime.timedelta(days=1)

    # データの取得
    stock_data = pdr.get_data_yahoo(brand, start_date, end_date, interval='1m')
    latest_price = stock_data['Close'].values.astype('float32')

#------------------------------------------------
#出力
#------------------------------------------------

    # 結果の表示
    st.write(f"予測価格({yosoku_date.strftime('%m/%d 02:00')}): {predicted_today[0][0]}")

    # 現在の価格を表示
    st.write(f"現在価格({end_date.strftime('%m/%d %H:%M')}): {latest_price[-1]}")
    
    # 現在の価格を表示
    st.write(f"差分:{latest_price[-1] - predicted_today[0][0]}")

#------------------------------------------------
#グラフの描画
#------------------------------------------------

    predicted = torch.from_numpy(predicted)
    y_test = torch.from_numpy(y_test)

    # 正規化解除
    predicted = scaler_price.inverse_transform(predicted.numpy())
    y_test = scaler_price.inverse_transform(y_test.numpy())

    # 予測結果と実際の結果の差分を計算
    diff = predicted - y_test

    # グラフの描画
    fig, ax = plt.subplots()
    ax.plot(y_test, label='Actual closing prices')
    ax.plot(predicted, label='Predicted closing prices')
    ax.set(title='Actual vs Predicted closing prices',
        xlabel='Time', ylabel='Price')
    ax.legend()
    # Matplotlibのグラフをウェブページに表示する
    st.pyplot(fig)

    # # グラフの描画
    # fig, ax = plt.subplots()
    # ax.plot(diff)
    # ax.set(title='Difference between predicted and actual closing prices',
    #     xlabel='Time', ylabel='Price difference')
    # # Matplotlibのグラフをウェブページに表示する
    # st.pyplot(fig)


