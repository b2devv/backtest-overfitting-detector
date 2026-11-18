# backtest-overfitting-detector
Backtest overfitting detector in Python. It generates 1,000 random trading rules (moving average, momentum, mean reversion, breakout), backtests them with costs and no lookahead, and picks the best. It then applies the Deflated Sharpe Ratio to show whether that winner is real skill or just luck from trying many strategies. Tested on 5 years of SPY.
