from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import json
import config
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

#Get the data
gotData = False 
while gotData == False:
		try:
			ts = TimeSeries(key=config.av_key, output_format = 'pandas')
			amazon_data, amazon_meta_data = ts.get_daily(symbol='AMZN', outputsize='full')
			gotData = True
		except:
			print('alpha vantage api error')

#Potting closing price
amazon_data['4. close'].plot()
plt.title('Closing Prices for AMZN Stock')
plt.show()

#Bolliger Bands

ti = TechIndicators(key=config.av_key, output_format = 'pandas')
b_data, b_meta_data = ti.get_bbands(symbol = 'AMZN')
b_data.plot()
plt.title('BBbands for AMZN')
plt.show()

## Related Securities

#Pull in other tech stocks

got_tech_data = False
while got_tech_data == False:
	try:
		google_data, google_meta_data = ts.get_daily(symbol='GOOG', outputsize='full')
		facebook_data, facebook_meta_data = ts.get_daily(symbol='FB', outputsize='full')
		got_tech_data = True
	except:
		print('alpha vantage api error')


#Pull in Walmart
got_walmart_data = False
while got_walmart_data == False:
	try:
		walmart_data, walmart_meta_data = ts.get_daily(symbol='WMT', outputsize='full')
		got_walmart_data = True
	except:
		print('alpha vantage api error')
#Merge the data
amazon_data['amazon_close'] = amazon_data['4. close']
google_data['google_close'] = google_data['4. close']
facebook_data['facebook_close'] = facebook_data['4. close']
walmart_data['walmart_close'] = walmart_data['4. close']

frames = [amazon_data['amazon_close'], \
		  google_data['google_close'], \
		  facebook_data['facebook_close'], \
		  walmart_data['walmart_close']]

combined_df = pd.concat(frames, axis=1)

combined_df.index = pd.to_datetime(combined_df.index)

print('combined df:')
print(combined_df.head())

print(type(combined_df.index))

#ultimately we want to analyze 2015 data. 
#to do this, pull from Dec. 2014 on so that we have moving averages on 1/1/15

combined_df = combined_df['2014-12-01':]
print(combined_df.head())

## Tech indicators

def get_technical_indicators(dataset):
	#moving averages:
	dataset['amazon_ma7'] = dataset['amazon_close'].rolling(window=7).mean()
	dataset['amazon_ma21'] = dataset['amazon_close'].rolling(window=21).mean()

	#Moving average convergence divergence
	dataset['amazon_26ema'] = pd.ewma(dataset['amazon_close'], span=26)
	dataset['amazon_12ema'] = pd.ewma(dataset['amazon_close'], span=26)
	dataset['amazon_macd'] = (dataset['amazon_12ema'] - dataset['amazon_26ema'])

	#Bollinger Bands

	dataset['amazon_20sd'] = pd.stats.moments.rolling_std(dataset['amazon_close'], 20)
	dataset['upper_band'] = dataset['amazon_ma21'] + (dataset['amazon_20sd']*2)
	dataset['lower_band'] = dataset['amazon_ma21'] - (dataset['amazon_20sd']*2)

	#Exponential moving average
	dataset['amazon_ema'] = dataset['amazon_close'].ewm(com=0.5).mean()

	#Create momentum
	dataset['amazon_momentum'] = dataset['amazon_close'] -1

	return dataset

new_amazon_df = get_technical_indicators(combined_df)

new_amazon_df = new_amazon_df['2015-01-01':]
print(new_amazon_df.head())

#fourier transform

amazon_df_FT = new_amazon_df[['amazon_close']]

close_fft = np.fft.fft(np.asarray(amazon_df_FT['amazon_close'].tolist()))
amazon_fft_df = pd.DataFrame({'fft': close_fft})
amazon_fft_df['absolute'] = amazon_fft_df['fft'].apply(lambda x: np.abs(x))
amazon_fft_df['angle'] = amazon_fft_df['fft'].apply(lambda x: np.angle(x)) 


print(amazon_fft_df.head())

#wavelets


items = deque(np.asarray(amazon_fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(amazon_fft_df)/2)))
plt.figure(figsize=(10,7), dpi=80)
plt.stem(items)
plt.title('Wavlets')
plt.show()