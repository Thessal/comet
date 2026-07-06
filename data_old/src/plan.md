the binary does the following task

* download 
https://data.londonstrategicedge.com/candles/stocks/aapl/1h/2024.csv.gz
https://data.londonstrategicedge.com/candles/stocks/nvda/1h/2024.csv.gz
https://data.londonstrategicedge.com/candles/stocks/tsla/1h/2024.csv.gz
https://data.londonstrategicedge.com/candles/stocks/msft/1h/2024.csv.gz
https://data.londonstrategicedge.com/candles/stocks/amzn/1h/2024.csv.gz

* align data into tables 
open.csv.gz # data file (n lines (newline separated), p columns (comma separated))
open.columns.csv.gz # instrument ids (aapl, nvda, or isin, bloomberg codes..) (single line, comma separated)
open.index.csv.gz # datetime (newline separated)
high.*
low.*
close.*
volume.*

* derived data
return.* # simple close / delay(close,1) - 1. for simplicity. it's liquid stocks anyway