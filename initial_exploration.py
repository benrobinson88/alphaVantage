#See https://github.com/RomelTorres/alpha_vantage/blob/develop/README.md

from alpha_vantage.timeseries import TimeSeries
import json
import config
from pprint import pprint
ts = TimeSeries(key=config.av_key, output_format = 'pandas')
data, meta_data = ts.get_daily('RUBI')

pprint(data.head(180))
