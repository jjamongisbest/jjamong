import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e])
train_y = np.array(data_y[0 : train_size])

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

# raw_df.close[-1] : dfy.close[-1] = x : pred_y[-1]

today_SEC_price = pred_y[-2]
tomorrow_SEC_price =pred_y[-1]

if today_SEC_price < tomorrow_SEC_price:
    upper_price = (tomorrow_SEC_price - today_SEC_price) / today_SEC_price * 100
    print("선택하신 종목은 다음 날",round(float(upper_price), 2), "% 상한가로 예상됩니다.")
elif today_SEC_price > tomorrow_SEC_price:
    lower_price = (today_SEC_price - tomorrow_SEC_price) / today_SEC_price * 100
    print("선택하신 종목은 다음 날",round(float(lower_price), 2), "% 하한가로 예상됩니다.")
else:
    print("선택하신 종목은 다음 날 주식 동결입니다.")

# 가격 예측에 어려움이 있어 추세를 파악하여 상한 및 하한을 예상해주기로 함
