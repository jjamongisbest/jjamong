from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
from Investar import Analyzer

mk = Analyzer.MarketDB()
stocks = input('종목을 입력하세요 : ')
raw_df = mk.get_daily_price(stocks, '2021-01-01', '2022-09-30')

window_size = 10
data_size = 5

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

dfx = raw_df[['open','high','low','volume', 'close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['close']]

x = dfx.values.tolist()
y = dfy.values.tolist()

data_x = []
data_y = []
window_size = 10
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

print(_x, "->", _y)

#훈련용 데이터셋
train_size = int(len(data_y) * 0.7)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])

#테스트용 데이터셋
test_size = len(data_y) - train_size
test_x = np.array(data_x[train_size : len(data_x)])
test_y = np.array(data_y[train_size : len(data_y)])

# 모델 생성
model = Sequential()
model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, data_size)))
model.add(Dropout(0.1))
model.add(LSTM(units=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, epochs=60, batch_size=30)
pred_y = model.predict(test_x)

# Visualising the results
plt.figure()
plt.xticks(np.arange(0,0))
plt.yticks(np.arange(0,0))
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

# 다음 날 주가의 상한/하한 예측하기
today_SEC_price = pred_y[-2]
tomorrow_SEC_price = pred_y[-1]

if today_SEC_price < tomorrow_SEC_price:
    upper_price = (tomorrow_SEC_price - today_SEC_price) / today_SEC_price * 100
    print("선택하신 종목은 다음 날",round(float(upper_price), 2), "% 상한가로 예상됩니다.")
elif today_SEC_price > tomorrow_SEC_price:
    lower_price = (today_SEC_price - tomorrow_SEC_price) / today_SEC_price * 100
    print("선택하신 종목은 다음 날",round(float(lower_price), 2), "% 하한가로 예상됩니다.")
else:
    print("선택하신 종목은 다음 날 주식 동결입니다.")
