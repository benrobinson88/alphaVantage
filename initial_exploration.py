from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import json
import config
from pprint import pprint
import matplotlib.pyplot as plt

#Get the data
ts = TimeSeries(key=config.av_key, output_format = 'pandas')
data, meta_data = ts.get_daily(symbol='AMZN', outputsize='full')

#Potting closing price
data['4. close'].plot()
plt.title('Closing Prices for AMZN Stock')
plt.show()

#Bolliger Bands

ti = TechIndicators(key=config.av_key, output_format = 'pandas')
data, meta_data = ti.get_bbands(symbol = 'AMZN')
data.plot()
plt.title('BBbands for AMZN')
plt.show()







