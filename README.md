# Crypto Trading Signals Generator

A simple system for generating cryptocurrency trading signals based on technical analysis while avoiding forward bias.

1. Gets the price data from the coingecko api. Also gets current market data.
2. Uses polynomial smoothing to smooth the price data and detect peaks and troughs.
4. Generates buy/sell signals based on ATH (All-Time High) prices, peaks and troughs using savitzky-golay find_peaks function.
5. Tracks token prices and major swings in price and alerts.
6. Important metrics are visualized in Plotly to aid decision making.

# Important Features

1. Minimal forward-looking bias in calculations (sell signal generated for time t set atmost at t+2 hours)
2. Data updated every hour.
3. Proper frequency handling (daily/hourly).
4. State management for portfolio tracking
5. Automated notifications via Telegram

# Sample Visualization

### Near token Price and Signals

![Near Price and Signals](./sample_images/near_token_sample.png)

### Oasis token Price and Signals

![Oasis Price and Signals](sample_images/oasis_token_sample.png)

# Environment Setup

1. Copy `.env.sample` to `.env`:
   ```bash
   cp .env.sample .env
   ```

2. Fill in your credentials in `.env`:
   - Get Telegram Bot Token from @BotFather
   - Get Telegram Chat ID from @userinfobot
   - Get CoinGecko API Keys from your CoinGecko account

3. Add token data
   - Go to Coingecko's historical price data page and get the entire daily price history. This is for bitcoin - https://www.coingecko.com/en/coins/bitcoin/historical_data?start=1900-12-14&end=2024-12-13
   - Download and add the file in the historical_data folder.
   - Update the TRACKED_TOKENS dictionary with the new token data.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
