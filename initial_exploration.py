#See https://github.com/RomelTorres/alpha_vantage/blob/develop/README.md

from alpha_vantage.timeseries import TimeSeries
import json
import config
ts = TimeSeries(key=config.av_key)
data, meta_data = ts.get_daily('RUBI')

print json.dumps(data)
