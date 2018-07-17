from alpha_vantage.timeseries import TimeSeries
import json
import config
from pprint import pprint
import matplotlib.pyplot as plt

ts = TimeSeries(key=config.av_key, output_format = 'pandas')
data, meta_data = ts.get_daily(symbol='RUBI', outputsize='full')
data['4. close'].plot()
plt.title('Closing Prices for RUBI Stock')
plt.show()



