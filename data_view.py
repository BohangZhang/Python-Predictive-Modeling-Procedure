import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ETH-USDT_data_new.csv", header=0)
#df = df.sort_values(by=['depth', 'bid1_price'])
df_unique = df.drop_duplicates(keep=False)

df_unique.depthTime = df_unique.depthTime.astype(int)
df_unique.bid1_price = df_unique.bid1_price.astype(np.float64)

df['spread'] = df.ask1_price - df.bid1_price
plt.plot(df.lastTrade_price)
df_unique = df_unique.sort_values(by='depthTime')



