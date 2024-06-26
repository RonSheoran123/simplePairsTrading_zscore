import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
import seaborn as sns

symbol_list = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
df = yf.download(
    symbol_list, 
    start='2014-01-01',end='2015-01-01'
)['Adj Close']

def find_coint_pairs(df):
    n=df.shape[1]
    p_matrix=np.zeros((n,n))
    keys=df.keys()
    pairs=[]
    for i in range(n):
        for j in range(i+1,n):
            s1=df[keys[i]]
            s2=df[keys[j]]
            result=coint(s1,s2)
            p_matrix[i][j]=result[1]
            if result[1]<0.05:
                pairs.append((keys[i],keys[j]))
    p_matrix[p_matrix==0]=np.nan
    return p_matrix, keys

pvalues, pairs=find_coint_pairs(df)

sns.heatmap(pvalues, xticklabels=symbol_list, yticklabels=symbol_list, cmap='RdYlGn_r', mask=((pvalues>0.05)))

print(coint(df.AMZN,df.GOOG)[1]) #0.032861650266896324

s1=df.GOOG
s2=df.AMZN

s1=sm.add_constant(s1)
results=sm.OLS(s2,s1).fit()
s1=s1.GOOG
b=results.params['GOOG']
spread=s2-b*s1

def zscore(series):
    return (series-series.mean())/np.std(series)

trades=pd.concat([zscore(spread),s2-b*s1],axis=1)
trades.columns=["signal","position"]

trades["side"]=0.0
trades.loc[trades.signal>1,"side"]=-1
trades.loc[trades.signal<-1,"side"]=1

df_merged=pd.concat([df['AMZN'],df['GOOG'],trades],ignore_index=True,axis=1)
df_merged.columns=['AMZN', 'GOOG', 'signal', 'position', 'side']
df_merged

money=0
count_s1=0
count_s2=0
for index, row in df_merged.iterrows():
    if row.side==1:
        money-=row.AMZN+b*row.GOOG
        count_s1+=1
        count_s2-=b
    elif row.side==-1:
        money+=row.AMZN-b*row.GOOG
        count_s1-=1
        count_s2+=b
    elif row.signal<0.1:
        money+=count_s1*row.AMZN + count_s2*row.GOOG
        count_s1=0
        count_s2=0

print(money)
