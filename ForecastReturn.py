#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Investar import Analyzer

mk = Analyzer.MarketDB()
stocks = list(input('종목을 입력하세요. 종목은 띄어쓰기로 구분합니다. (예시: 삼성전자 SK하이닉스 현대자동차 NAVER)\n:').split())
a = float(input('금액 설정:'))
df = pd.DataFrame()
for s in stocks:
    df[s] = mk.get_daily_price(s, '2019-01-01', '2022-08-21')['close']
  
daily_ret = df.pct_change() 
annual_ret = daily_ret.mean() * 252
daily_cov = daily_ret.cov()
annual_cov = daily_cov * 252

port_ret = [] 
port_risk = [] 
port_weights = []
sharpe_ratio = [] 

for _ in range(20000): 
    weights = np.random.random(len(stocks)) 
    weights /= np.sum(weights) 

    returns = np.dot(weights, annual_ret) 
    risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights))) 

    port_ret.append(returns) 
    port_risk.append(risk) 
    port_weights.append(weights)
    sharpe_ratio.append(returns/risk)  # ①

portfolio = {'Returns': port_ret, 'Risk': port_risk, 'Sharpe': sharpe_ratio}
for i, s in enumerate(stocks): 
    portfolio[s] = [weight[i] for weight in port_weights] 
df = pd.DataFrame(portfolio) 
df = df[['Returns', 'Risk', 'Sharpe'] + [s for s in stocks]]  # ②

max_sharpe = df.loc[df['Sharpe'] == df['Sharpe'].max()]  # ③
min_risk = df.loc[df['Risk'] == df['Risk'].min()]  # ④
max_sharpe_returns = float(max_sharpe['Returns'])
max_sharpe_risk = 100 * float(max_sharpe['Risk'])
min_risk_returns = float(min_risk['Returns'])
min_risk_risk = 100 * float(min_risk['Risk'])
b = max_sharpe_returns * a
c = min_risk_returns * a

print('\n주의 : 앞의 포트폴리오 결과와 약간의 오차가 발생할 수 있습니다.')
print('\n1. 최대 수익 (큰 리스크를 감수하여 최대 수익을 얻습니다)')
print(int(a), '원을 투자하시면', int(max_sharpe_risk),'%의 리스크를 겪으면서', int(b), '원의 최대 수익을 얻을 수 있습니다.')
print('투자 방법 :')
for i in range(len(stocks)):
    print(stocks[i-1], '종목에', int(a * float(max_sharpe[stocks[i-1]])) , '원 투자')

print('\n2. 최소 리스크 (최소 리스크를 감수하여 적정 수익을 얻습니다)')
print(int(a), '원을 투자하시면', int(min_risk_risk),'%의 최소 리스크를 겪으면서', int(c), '원의 수익을 얻을 수 있습니다.')
print('투자 방법 :')
for i in range(len(stocks)):
    print(stocks[i-1], '종목에', int(a * float(min_risk[stocks[i-1]])) , '원 투자')