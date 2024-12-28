# %%
# --- To Do ---
# Still issues with the price smoothing - look at maple, clearpool
# You are actually using polyval instead of savgol_filter. Assess next steps.

# %%

# --- Imports ---

import tempfile
import numpy as np
import requests
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import time
import diskcache as dc
from IPython.display import display as dp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import os

from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler


from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# %%

# -- Setup --

pd.options.mode.chained_assignment = None  # default is 'warn'

if not load_dotenv(".env"):
    logging.error(f"Failed to load .env file from")
    raise RuntimeError(f"Failed to load .env file from")
logging.info(f"Successfully loaded .env file from")

# %%

# -- Logging --


def setup_logging():
    """Configure the logging system"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup file handler with rotation
    file_handler = RotatingFileHandler(
        filename=log_dir / "crypto_signals.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    file_handler.setFormatter(formatter)

    # Setup handlers for different components
    loggers = {
        "api": logging.getLogger("crypto_signals.api"),
        "data": logging.getLogger("crypto_signals.data"),
        "signal": logging.getLogger("crypto_signals.signal"),
        "notification": logging.getLogger("crypto_signals.notification"),
        "system": logging.getLogger("crypto_signals.system"),
    }

    # Configure all loggers
    for logger in loggers.values():
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.propagate = False

    return loggers


# Initialize loggers
loggers = setup_logging()

# %%

# -- Global Variables --

# Configuration
API_KEYS = [os.getenv("CG_API_KEY_1"), os.getenv("CG_API_KEY_2"), os.getenv("CG_API_KEY_3")]

# Token tracking settings
TRACKED_TOKENS = {
    "near": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "oasis-network": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "ix-swap": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "empyreal": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "maple": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "clearpool": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "simmi-token": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
    "ondo-finance": {"track_buy": False, "track_sell": True, "peak_limit": 0.99},
}

# %%

# Telegram Configuration


class NotificationManager:
    def __init__(self):
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    def get_fear_greed_index(self):
        """Get the fear and greed index and return formatted message"""

        data = requests.get("https://api.alternative.me/fng/?limit=1").json()["data"][0]
        value = int(data["value"])
        timestamp = pd.to_datetime(int(data["timestamp"]), unit="s")
        classification = data["value_classification"]
        assert "time_until_update" in data

        return f"Fear & Greed Index ({add_india_offset(timestamp)}):\nValue: {value}\nClassification: {classification}"

    def get_stats(self):
        """Get stats for a token"""

        mcap_df = get_market_cap_data()

        stats = {}
        if not mcap_df.empty:
            stats = {
                "alt_market_cap": round(mcap_df["alt_market_cap"].iloc[-1] / 1e12, 3),  # Convert to trillions
                "alt_market_cap_pct": round(mcap_df["alt_market_cap_pct"].iloc[-1], 2),
                "total_volume_pct": round(mcap_df["total_volume_pct"].iloc[-1], 2),
                "alt_dominance_pct": round(mcap_df["alt_dominance_pct"].iloc[-1], 2),
            }

            message = f"""Market Stats ({mcap_df.index[-1]}):\n\
Alt Market Cap: ${stats['alt_market_cap']}T\n\
Alt Market Cap pct: {stats['alt_market_cap_pct']}%\n\
Total Volume pct: {stats['total_volume_pct']}%\n\
Alt Dominance pct: {stats['alt_dominance_pct']}%"""
            return message
        return "No market data available for today"

    def null_values_present(self, token, num_nulls):
        """Send Telegram notification about null values in token data"""
        message = f"Null values present: {token} has {num_nulls} null values in its price data"

        # Send message via Telegram bot
        url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": self.TELEGRAM_CHAT_ID, "text": message}
        try:
            requests.post(url, params=params)
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")
            print(message)  # Fallback to console

    def send_custom_message(self, message):
        """Send a custom message to the Telegram chat"""
        url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": self.TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, params=params)

    def send_price_alert_with_stats(self, token, message, df, send_stats=True, send_fig=True):
        logger = loggers["notification"]

        try:
            full_message = message + "\n"
            if send_stats:
                stats = self.get_stats()
                fear_greed = self.get_fear_greed_index()
                full_message += stats + "\n\n" + fear_greed

            # Send combined text message
            self.send_custom_message(full_message)

            if send_fig:
                # Create plotly figure
                fig = plot_portfolio_signals(df, token)

                # Convert to HTML with full plotly functionality
                chart_html = fig.to_html(include_plotlyjs="cdn", config={"scrollZoom": True, "displayModeBar": True})

                # Save HTML to temporary file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                    f.write(chart_html)
                    temp_path = f.name

                # Send HTML file via Telegram
                url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendDocument"
                with open(temp_path, "rb") as f:
                    files = {"document": f}
                    data = {"chat_id": self.TELEGRAM_CHAT_ID}
                    requests.post(url, files=files, data=data)

                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Failed to send price alert for {token}: {str(e)}", exc_info=True)
            # Clean up temp file


df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

# NotificationManager().send_price_alert_with_stats("empyreal", "test", df, send_stats=True, send_fig=True)
# %%

cache.clear()
# %%
cache = dc.Cache("cache_directory")  # This will store the cache in a folder called 'cache_directory'

notifmanager = NotificationManager()


class KeyRotator:
    def __init__(self, keys: list):
        self.keys = keys
        self.current = 0
        self.limited_keys = set()

    def next_key(self):
        """Get next available key, cycling through all keys"""
        # Try each key once
        for _ in range(len(self.keys)):
            key = self.keys[self.current]
            self.current = (self.current + 1) % len(self.keys)
            if key not in self.limited_keys:
                return key

        # All keys are limited
        return None

    def mark_limited(self, key):
        """Mark a key as rate limited"""
        self.limited_keys.add(key)

    def reset_limits(self):
        """Reset all rate limits after waiting period"""
        self.limited_keys.clear()


def api_rate_limit_call(url, use_cache=True, cg=True, max_retries=5, *args, **kwargs):
    logger = loggers["api"]

    # Return cached response if available
    if use_cache and (cached := cache.get(url)):
        return cached

    # Initialize key rotator if needed
    if not hasattr(api_rate_limit_call, "rotator"):
        api_rate_limit_call.rotator = KeyRotator(API_KEYS)

    retry_delay = 5.1  # Initial delay in seconds
    for attempt in range(max_retries):
        # Try up to max_retries cycles
        for cycle in range(len(API_KEYS)):
            # Reset rate limits at start of each cycle
            api_rate_limit_call.rotator.reset_limits()

            key = api_rate_limit_call.rotator.next_key()
            if key is None:
                break  # All keys are limited, proceed to wait

            try:
                # Construct URL and headers
                final_url = f"{url}?x_cg_demo_api_key={key}" if cg else url
                headers = {"x_cg_pro_api_key": key}

                response = requests.get(final_url, *args, **kwargs, headers=headers)

                if response.status_code == 200:
                    json_response = response.json()
                    if use_cache:
                        cache.set(url, json_response, expire=14 * 24 * 60 * 60)
                    return json_response

                if response.status_code == 429:  # Rate limit hit
                    api_rate_limit_call.rotator.mark_limited(key)
                    continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                error_msg = f"API request failed with key {key}: {e}"
                logger.error(error_msg)
                notifmanager.send_custom_message(error_msg)
                # For non-rate-limit errors, mark key as limited and try next

                api_rate_limit_call.rotator.mark_limited(key)

        # If all keys are rate limited, wait before retrying
        time.sleep(retry_delay)
        retry_delay *= 1.5  # Exponential backoff
        logger.info(f"All keys exhausted. Waiting {retry_delay}s before next attempt")

    # After max_retries attempts, raise error
    error_msg = "API request failed after multiple retries due to rate limiting."
    notifmanager.send_custom_message(error_msg)
    raise RuntimeError(error_msg)


# %%
class CryptoDataManager:
    def __init__(self, data_file_path="saved_data/token_data.csv"):
        self.data_file_path = data_file_path
        self.main_df = self._load_data()
        self.col_suffixes = [
            "price",
            "market_cap",
            "total_volume",
            "price_scaled",
            "total_volume_minmax",
            "smooth",
            "peaks",
            "portfolio_signal",
            "state",
        ]
        self.logger = loggers["data"]

    def _load_data(self):
        """Load existing CSV or create a new DataFrame."""
        if os.path.exists(self.data_file_path):
            return pd.read_csv(self.data_file_path, index_col=0, parse_dates=True)
        return pd.DataFrame()

    def save_data(self):
        """Save current data state to CSV."""
        self.main_df.to_csv(self.data_file_path)

    def get_token_columns(self, token):
        """Return column names for a specific token."""
        return [f"{token}_{suffix}" for suffix in self.col_suffixes]

    def remove_columns(self, tokens):
        """Keep only needed columns for the given tokens."""
        needed_cols = [col for token in tokens for col in self.get_token_columns(token)]
        cols_to_remove = [col for col in self.main_df.columns if col not in needed_cols]
        self.main_df.drop(columns=cols_to_remove, inplace=True)
        self.logger.info(f"Removed columns: {cols_to_remove}")

    def check_token_status(self, token):
        """Check if token exists and get the last valid entry."""
        token_cols = self.get_token_columns(token)
        if all(col in self.main_df.columns for col in token_cols):
            token_data = self.main_df[token_cols].dropna(how="all")
            if not token_data.empty:
                return True, token_data.index[-1]
        return False, None

    def _process_api_data(self, data):
        """Process raw API data into a DataFrame."""
        timestamps = [ts[0] for ts in data.get("prices", [])]
        df = pd.DataFrame(
            {
                "price": [price[1] for price in data.get("prices", [])],
                "market_cap": [mc[1] for mc in data.get("market_caps", [])],
                "total_volume": [tv[1] for tv in data.get("total_volumes", [])],
            },
            index=pd.to_datetime(timestamps, unit="ms"),
        )
        return df

    def load_historical_data(self, token):
        """Load historical data from CSV."""
        file_path = os.path.join("historical data", f"{token}-usd-max.csv")
        if not os.path.exists(file_path):
            self.logger.error(f"File {file_path} not found.")
            return None

        df = pd.read_csv(file_path, parse_dates=["snapped_at"], index_col="snapped_at")
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="D"))
        df = df.interpolate(method="linear")
        self.logger.info("Historical data loaded")
        return df

    def fetch_hourly_data(self, token, start_date, end_date=None):
        """Fetch hourly data for a specified date range with overlapping intervals."""
        end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.utcnow().tz_convert(None).ceil("H")
        start_date = pd.to_datetime(start_date)

        token_df = pd.DataFrame()
        current_start = start_date
        total_days = (end_date - start_date).days

        with tqdm(total=total_days, desc=f"Fetching {token} data") as pbar:
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=90), end_date)
                url = (
                    f"https://api.coingecko.com/api/v3/coins/{token}/market_chart/range"
                    f"?vs_currency=usd&from={int(current_start.timestamp())}&to={int(current_end.timestamp())}"
                )
                data = api_rate_limit_call(url)

                if data:
                    temp_df = self._process_api_data(data)
                    token_df = pd.concat([token_df, temp_df])

                pbar.update((current_end - current_start).days)
                current_start = current_end

        if token_df.empty:
            return None

        token_df = token_df.sort_index()
        token_df.index = pd.to_datetime(token_df.index).round("H")
        token_df = token_df.groupby(token_df.index).mean()
        token_df = token_df.reindex(pd.date_range(start=token_df.index.min(), end=token_df.index.max(), freq="H"))
        return token_df.interpolate(method="linear")

    def update_token_data(self, token):
        """Update or add token data."""
        exists, last_valid_date = self.check_token_status(token)
        token_cols = self.get_token_columns(token)

        if exists:
            self.logger.info(f"Previous data detected for token {token}")
            start_date = min(
                last_valid_date,
                pd.Timestamp.utcnow().tz_convert(None).ceil("H") - timedelta(days=3),
            )
            data = self.fetch_hourly_data(token, start_date=start_date)
        else:
            self.logger.info(f"No previous data detected for token {token}")
            historical_data = self.load_historical_data(token)
            hourly_data = self.fetch_hourly_data(
                token,
                start_date=pd.Timestamp.utcnow().tz_convert(None).ceil("H") - timedelta(days=365),
            )
            data = self._integrate_data(historical_data, hourly_data)

        if data is None:
            self.logger.error(f"No data found in combined historical and hourly data for token {token}")
            return False

        if self.main_df.empty:
            self.main_df = pd.DataFrame(
                {
                    token_cols[0]: data["price"],
                    token_cols[1]: data["market_cap"],
                    token_cols[2]: data["total_volume"],
                },
                index=data.index,
            )
        else:
            for col in token_cols:
                if col not in self.main_df.columns:
                    self.main_df[col] = np.nan

            self.main_df = self.main_df.combine_first(
                data.rename(
                    columns={
                        "price": token_cols[0],
                        "market_cap": token_cols[1],
                        "total_volume": token_cols[2],
                    }
                )
            )

        return True

    def _integrate_data(self, historical_data, hourly_data):
        """Integrate historical and hourly data."""
        if historical_data is None:
            return hourly_data
        if hourly_data is None:
            return historical_data

        combined = pd.concat([historical_data, hourly_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self.logger.info(f"Data integrated for token")
        return combined

    def update_multiple_tokens(self, tokens):
        """Update multiple tokens at once."""
        self.remove_columns(tokens)
        for token in tokens:
            self.logger.info(f"\nProcessing {token}...")
            if self.update_token_data(token):
                self.logger.info(f"Successfully updated {token}")
            else:
                self.logger.error(f"Failed to update {token}")

        self.save_data()


# manager = CryptoDataManager()
# tokens = ["near", "oasis-network", "ix-swap", "empyreal", "maple", "clearpool", "simmi-token", "ondo-finance"]

# # tokens = ['empyreal', 'fideum']
# manager.update_multiple_tokens(tokens)


# %%


def add_india_offset(dt: datetime) -> datetime:
    """
    Add 5 hours and 30 minutes to a datetime object (IST offset).

    Args:
        dt: Input datetime object

    Returns:
        datetime: Datetime with 5:30 hours added
    """
    return dt + timedelta(hours=5, minutes=30)


# %%


def notify_price_alerts(df: pd.DataFrame):
    """
    Check price changes over multiple timeframes and send alerts if thresholds are exceeded in either direction.
    Uses checkpoints to avoid duplicate alerts.

    Args:
        df: DataFrame containing token price data with datetime index

    Timeframes and thresholds:
        7d: ±50% change
        1d: ±30% change
        1h: ±15% change
    """
    # Define timeframes and thresholds
    timeframe_thresholds = {"7d": 0.50, "1d": 0.30, "1h": 0.15}  # ±50% change  # ±30% change  # ±15% change

    # Load checkpoints from file if exists
    checkpoint_file = "saved_data/price_alert_checkpoints.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoints = json.load(f)
            # Convert string timestamps back to datetime
            for token in checkpoints:
                for timeframe in checkpoints[token]:
                    if checkpoints[token][timeframe]:
                        checkpoints[token][timeframe] = pd.to_datetime(checkpoints[token][timeframe])
    else:
        checkpoints = {}

    notifmanager = NotificationManager()
    # Get list of tokens from price columns
    tokens = [col.split("_")[0] for col in df.columns if col.endswith("_price")]

    for token in tokens:
        if token not in checkpoints:
            checkpoints[token] = {tf: None for tf in timeframe_thresholds.keys()}

        price_col = f"{token}_price"
        current_price = df[price_col].iloc[-1]
        current_time = df.index[-1]

        # Check each timeframe
        for timeframe, threshold in timeframe_thresholds.items():
            # Convert timeframe to offset
            if timeframe.endswith("d"):
                offset = pd.Timedelta(days=int(timeframe[:-1]))
            elif timeframe.endswith("h"):
                offset = pd.Timedelta(hours=int(timeframe[:-1]))

            # Determine start time based on checkpoint
            checkpoint = checkpoints[token][timeframe]
            timeframe_start = current_time - offset

            # Use more recent of checkpoint and timeframe_start
            if checkpoint and checkpoint > timeframe_start:
                start_time = checkpoint
            else:
                start_time = timeframe_start
                # Drop checkpoint if it's older than timeframe window
                if checkpoint and checkpoint < timeframe_start:
                    checkpoints[token][timeframe] = None

            # Get price at start time
            start_price = df[df.index <= start_time][price_col].iloc[-1]

            # Calculate price change
            pct_change = (current_price - start_price) / start_price

            # Send alert if threshold exceeded in either direction
            if abs(pct_change) >= threshold:
                # Determine emoji and direction based on positive/negative change
                if pct_change > 0:
                    emoji = "🚀"
                    direction = "+"
                else:
                    emoji = "📉"
                    direction = "-"

                message = (
                    f"{emoji} Price Alert: {token}\n"
                    f"Timeframe: {timeframe}\n"
                    f"{direction} {abs(pct_change):.1%}\n"
                    f"From price at {add_india_offset(start_time)}: ${start_price:.4f}\n"
                    f"To price at {add_india_offset(df.index[-1])}: ${current_price:.4f}"
                )
                notifmanager.send_custom_message(message=message)

                # Update checkpoint after alert
                checkpoints[token][timeframe] = current_time

    # Save updated checkpoints
    with open(checkpoint_file, "w") as f:
        # Convert datetime to string for JSON serialization
        json_checkpoints = {}
        for token in checkpoints:
            json_checkpoints[token] = {}
            for timeframe in checkpoints[token]:
                if checkpoints[token][timeframe]:
                    json_checkpoints[token][timeframe] = checkpoints[token][timeframe].isoformat()
                else:
                    json_checkpoints[token][timeframe] = None

        json.dump(json_checkpoints, f, indent=4)


# df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)
# notify_price_alerts(df)


# %%
def validate_token_price_nulls(df: pd.DataFrame) -> bool:
    """
    Validate that all token price columns have no more than 2 null values in their last valid range.

    Args:
        df: DataFrame containing token price data with columns named {token}_price

    Returns:
        bool: True if validation passes, False otherwise

    Raises:
        AssertionError: If any token has more than 2 null values in its valid range
    """

    notifmanager = NotificationManager()

    # Get list of token names from price columns
    token_names = [f"{col.split('_')[0]}" for col in df.columns if col.endswith("_price")]

    for token in token_names:
        price_col = f"{token}_price"
        first_valid_index = df[price_col].first_valid_index()
        num_nulls = df.loc[first_valid_index:][price_col].isnull().sum()

        # For tokens with more than 4 nulls, check for hourly data transition issues
        if num_nulls > 50:
            # Get the token's price data
            price_series = df[price_col]

            # Calculate time differences between consecutive valid data points
            valid_data = price_series.dropna()
            time_diffs = valid_data.index.to_series().diff()

            # Daily data will have ~24 hour differences, hourly data will have ~1 hour differences
            # Add buffer for slight variations
            hourly_mask = time_diffs == pd.Timedelta(hours=1)
            daily_mask = time_diffs >= pd.Timedelta(hours=23)

            if hourly_mask.any() and daily_mask.any():
                # Find the transition point from daily to hourly data
                # This is where we consistently start seeing hourly differences
                for i in range(len(hourly_mask) - 3):  # Check for 3 consecutive hourly points
                    if all(hourly_mask.iloc[i : i + 3]):  # Found transition
                        # Get the index BEFORE the first hourly point to capture transition
                        hourly_start = valid_data.index[i - 1] if i > 0 else valid_data.index[i]

                        # print(hourly_start)

                        # Keep only daily data before transition
                        before_hourly = df.loc[:hourly_start]
                        daily_data = before_hourly[before_hourly.index.hour == 0]

                        # print(bef)

                        # Combine with all data after transition point
                        after_hourly = df.loc[hourly_start:]

                        df = pd.concat([daily_data, after_hourly])

                        # Update the original dataframe
                        df.to_csv("saved_data/token_data.csv")

                        notifmanager.send_custom_message(
                            f"Fixed daily/hourly transition for {token}. Hourly data starts from {hourly_start}"
                        )
                        break

        if num_nulls > 2:
            notifmanager.null_values_present(token, num_nulls)

    return True


# df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)
# validate_token_price_nulls(df)


# %%
def get_market_cap_data(save_path="market_cap_history.csv", to_save=True):
    """
    Fetch and maintain market cap data with proper handling of daily historical
    and hourly recent data.
    """

    def process_dataframe(df):
        """Process raw dataframe into final format"""
        df = (
            df.assign(
                timestamp=lambda x: pd.to_datetime(pd.to_numeric(x["timestamp"]), unit="s"),
                alt_market_cap=lambda x: x[["ethValue", "stableValue", "otherValue"]].sum(axis=1),
            )
            .set_index("timestamp")
            .sort_index()
        )
        df["alt_dominance"] = df["alt_market_cap"] * 100 / (df["btcValue"] + df["alt_market_cap"])

        # Add expanding percentile versions
        for col in ["volume", "alt_market_cap", "alt_dominance"]:
            df[f"{col}_pct"] = df[col].expanding().rank(pct=True) * 100

        return df.rename(
            columns={
                "marketCap": "total_market_cap",
                "volume": "total_volume",
                "btcValue": "btc_market_cap",
                "volume_pct": "total_volume_pct",
            }
        ).drop(columns=["ethValue", "stableValue", "otherValue"])

    file_path = Path(save_path)

    # Check if file exists and is up to date
    if file_path.exists():
        saved_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if datetime.now(timezone.utc).replace(tzinfo=None) < saved_data.index[-1] + pd.Timedelta(hours=1):
            return saved_data

    # Fetch new data
    all_time = api_rate_limit_call(
        "https://api.coinmarketcap.com/data-api/v4/global-metrics/quotes/historical?convertId=2781&range=All",
        cg=False,
        use_cache=False,
    )["data"]["points"]

    year = api_rate_limit_call(
        "https://api.coinmarketcap.com/data-api/v4/global-metrics/quotes/historical?convertId=2781&range=1y",
        cg=False,
        use_cache=False,
    )["data"]["points"]

    month = api_rate_limit_call(
        "https://api.coinmarketcap.com/data-api/v4/global-metrics/quotes/historical?convertId=2781&range=30d",
        cg=False,
        use_cache=False,
    )["data"]["points"]

    # Process data
    all_df = process_dataframe(pd.DataFrame(all_time))
    year_df = process_dataframe(pd.DataFrame(year))
    month_df = process_dataframe(pd.DataFrame(month))

    # Round timestamps appropriately
    all_df.index = all_df.index.round("D")
    all_df = all_df.iloc[:-1]
    year_df.index = year_df.index.round("D")
    year_df = year_df.iloc[:-1]
    month_df.index = month_df.index.round("H")
    month_df = month_df.iloc[:-1]

    # Combine and deduplicate
    updated_df = pd.concat([all_df, year_df, month_df])
    updated_df = updated_df[~updated_df.index.duplicated(keep="last")].sort_index()

    if to_save:
        updated_df.to_csv(file_path)

    return updated_df


# mcap_df = get_market_cap_data()
# mcap_df


# %%
def get_fear_greed_index() -> pd.DataFrame:
    """
    Fetches and processes the Fear & Greed Index data from alternative.me API.

    Returns:
        pd.DataFrame: DataFrame containing fear & greed index values indexed by timestamp
    """
    alt_fgindex = pd.DataFrame(
        api_rate_limit_call("https://api.alternative.me/fng/?limit=0", cg=False, use_cache=False).json()["data"]
    )
    alt_fgindex["timestamp"] = pd.to_datetime(alt_fgindex["timestamp"], unit="s")

    alt_fgindex = alt_fgindex.set_index("timestamp").sort_index()
    alt_fgindex["value"] = alt_fgindex["value"].astype(int)

    return alt_fgindex


# get_fear_greed_index()
# api_rate_limit_call("https://api.alternative.me/fng").json()['data']


# %%
# -- Portfolio Signals (triple peak) --


def calculate_portfolio_signals_for_token(
    df: pd.DataFrame, complete_df: pd.DataFrame, token: str, initial_state: dict = None, send_notifications: bool = True
):
    """Calculate portfolio signals for a single token with state management"""
    logger = loggers["signal"]

    price_col = f"{token}_price"
    price_scaled_col = f"{token}_price_scaled"
    peak_col = f"{token}_peaks"
    signal_col = f"{token}_portfolio_signal"
    state_col = f"{token}_state"

    notifmanager = NotificationManager()

    # Initialize result DataFrame with correct dtypes
    result = pd.DataFrame(index=df.index)
    result[signal_col] = pd.Series(0.0, index=df.index, dtype=np.float64)
    result[state_col] = pd.Series(None, index=df.index, dtype="object")

    # Get peak_limit from TRACKED_TOKENS or use default
    peak_limit = TRACKED_TOKENS.get(token, {}).get("peak_limit", 0.99)

    # Initialize or load state with new last_ath_trigger field
    if initial_state is None:
        logger.debug(f"Initializing new state for token: {token}")
        state = {
            "last_ath_price": 0,
            "last_ath_trigger": 0,  # New field for tracking trigger price
            "last_ath_datetime": None,
            "peak_count": 0,
            "ath_triggered": False,
        }
    else:
        try:
            state = {
                "last_ath_price": float(initial_state.get("last_ath_price", 0)),
                "last_ath_trigger": float(initial_state.get("last_ath_trigger", 0)),
                "last_ath_datetime": initial_state.get("last_ath_datetime", None),
                "peak_count": int(initial_state.get("peak_count", 0)),
                "ath_triggered": bool(initial_state.get("ath_triggered", False)),
            }
            logger.debug(f"Loaded existing state for token: {token}")
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error initializing state for {token}: {str(e)}", exc_info=True)
            state = {
                "last_ath_price": 0,
                "last_ath_trigger": 0,
                "last_ath_datetime": None,
                "peak_count": 0,
                "ath_triggered": False,
            }

    # Get valid data range
    first_valid = df[price_col].first_valid_index()
    last_valid = df[price_col].last_valid_index()

    if first_valid is None or last_valid is None:
        return result

    # Process each point
    for i in df[first_valid:last_valid].index:
        # Get current values, handling potential NaN/None
        current_price = df.at[i, price_col]
        current_peak = df.at[i, peak_col]
        current_scaled_price = df.at[i, price_scaled_col]

        if pd.isna(current_price) or pd.isna(current_peak) or pd.isna(current_scaled_price):
            continue

        # Get latest valid values
        latest_price = df.at[last_valid, price_col]
        latest_scaled_price = df.at[last_valid, price_scaled_col]
        latest_datetime = last_valid

        LATEST_STATS_STR = (
            f"Latest Datetime: {latest_datetime}\n"
            f"Latest Price: ${latest_price:.4f}\n"
            f"Latest Scaled Price: {latest_scaled_price:.4f}\n"
        )

        # Reset conditions
        if current_scaled_price <= 20 and state["peak_count"] > 0:  # low_price_reset
            state.update(
                {
                    "last_ath_price": 0,
                    "last_ath_trigger": 0,  # Reset trigger price too
                    "last_ath_datetime": None,
                    "peak_count": 0,
                    "ath_triggered": False,
                }
            )

            if send_notifications:
                notifmanager.send_custom_message(f"ATH Reset: {token}")

        # ATH detection
        if (
            current_scaled_price >= 99.5
            and not state["ath_triggered"]
            and current_price > state["last_ath_trigger"] * peak_limit  # Compare with last trigger price
        ):
            state["last_ath_price"] = current_price
            state["last_ath_datetime"] = i
            state["ath_triggered"] = True

            if send_notifications and TRACKED_TOKENS.get(token, {}).get("track_sell", False) == True:
                # Create a temporary DataFrame with updated signals for plotting
                temp_df = complete_df.copy()
                temp_df.update(result)  # Merge the new signals into the complete DataFrame

                message = (
                    f"⚠️ ATH Triggered: {token}\n\n"
                    f"Triggered at: {i}\n"
                    f"Price: ${current_price:.4f}\n"
                    f"Peak Count: {state['peak_count']}\n"
                    f"{LATEST_STATS_STR}"
                )
                notifmanager.send_price_alert_with_stats(
                    token,
                    message,
                    temp_df,  # Use the temporary DataFrame with updated signals
                    send_stats=True,
                )

        # Update last_ath_price when peak not yet triggered
        if state["ath_triggered"] and current_peak != 1 and current_price > state["last_ath_price"]:
            state["last_ath_price"] = current_price
            state["last_ath_datetime"] = i

        # Execute sell at confirmed peak
        if state["ath_triggered"] and current_peak == 1:
            # Generate sell signal
            state["peak_count"] += 1
            state["ath_triggered"] = False
            state["last_ath_trigger"] = current_price  # Record the trigger price
            result.at[i, signal_col] = -1

            if send_notifications and TRACKED_TOKENS.get(token, {}).get("track_sell", False) == True:
                # Create a temporary DataFrame with updated signals for plotting
                temp_df = complete_df.copy()
                temp_df.update(result)  # Merge the new signals into the complete DataFrame

                message = (
                    f"🚨 Sell Signal: {token}\n\n"
                    f"Triggered at: {i}\n"
                    f"Triggered Price: ${current_price:.4f}\n"
                    f"Triggered Normalized Price: {current_scaled_price:.4f}\n"
                    f"\n"
                    f"Last ATH Price: ${state['last_ath_price']:.4f}\n"
                    f"Last ATH Trigger: ${state['last_ath_trigger']:.4f}\n"
                    f"Last ATH Datetime: {state['last_ath_datetime']}\n"
                    f"peak_count: {state['peak_count']} \n"
                    f"{LATEST_STATS_STR}"
                )
                notifmanager.send_price_alert_with_stats(
                    token,
                    message,
                    temp_df,  # Use the temporary DataFrame with updated signals
                    send_stats=True,
                )

        # Buy signal calculation
        if current_scaled_price <= 20 and current_peak == -1:  # low_price_buy_start
            buy_pct = min(1.0, (20 - current_scaled_price) / (20 - 5))
            result.at[i, signal_col] = buy_pct

            if send_notifications and TRACKED_TOKENS.get(token, {}).get("track_buy", False) == True:
                # Create a temporary DataFrame with updated signals for plotting
                temp_df = complete_df.copy()
                temp_df.update(result)  # Merge the new signals into the complete DataFrame

                message = (
                    f"💰 Buy Signal: {token}\n\n"
                    f"Triggered at: {i}\n"
                    f"Triggered Price: ${current_price:.4f}\n"
                    f"Triggered Normalized Price: {current_scaled_price:.4f}\n"
                    f"Buy_pct: {buy_pct:.2f}\n"
                    f"\n"
                    f"{LATEST_STATS_STR}"
                )
                notifmanager.send_price_alert_with_stats(
                    token,
                    message,
                    temp_df,  # Use the temporary DataFrame with updated signals
                    send_stats=True,
                )

        # Store state as string representation
        result.at[i, state_col] = str(state)

    return result


# %%
def plot_portfolio_signals(df: pd.DataFrame, token: str, show_plot: bool = True):
    """
    Interactive Plotly plot showing price, signals, and portfolio actions
    """
    # Get valid data range
    mask = df[f"{token}_price"].first_valid_index()
    plot_df = df.loc[mask:].copy()

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f"{token} Price and Actions", "Portfolio Signal"),
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[f"{token}_price"],
            name="Price",
            line=dict(color="blue", width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Price scaled and smooth lines
    for col, name in [(f"{token}_price_scaled", "Price Scaled"), (f"{token}_smooth", "Smooth Price")]:
        if col in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[col],
                    name=name,
                    line=dict(color="red", width=1),
                    opacity=0.7,
                ),
                secondary_y=True if "scaled" in col else False,
                row=1,
                col=1,
            )

    # Get buy/sell points from action column
    if f"{token}_portfolio_signal" in plot_df.columns:
        actions = plot_df[plot_df[f"{token}_portfolio_signal"] != 0]
        buys = actions[actions[f"{token}_portfolio_signal"] == 1]
        sells = actions[actions[f"{token}_portfolio_signal"] == -1]

        # Add buy/sell markers
        for points, color, symbol, name in [
            (buys, "green", "triangle-up", "Buy"),
            (sells, "red", "triangle-down", "Sell"),
        ]:
            if not points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=points.index,
                        y=points[f"{token}_price"],
                        mode="markers+text",
                        marker=dict(symbol=symbol, size=15, color=color),
                        text=[f"{abs(val*100):.0f}%" for val in points[f"{token}_portfolio_signal"]],
                        textposition="top center" if color == "green" else "bottom center",
                        name=name,
                        hovertemplate="Date: %{x}<br>Price: %{y}<br>Action: %{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    # Add peaks and troughs
    if f"{token}_peaks" in plot_df.columns:
        for value, color, name in [(1, "red", "Peak"), (-1, "green", "Trough")]:
            points = plot_df[plot_df[f"{token}_peaks"] == value]
            if not points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=points.index,
                        y=points[f"{token}_price"],
                        mode="markers",
                        marker=dict(symbol="circle", size=6, color=color),
                        name=name,
                    ),
                    row=1,
                    col=1,
                )

    # Portfolio signal
    if f"{token}_portfolio_signal" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[f"{token}_portfolio_signal"],
                name="Portfolio Signal",
                line=dict(color="purple", width=1.5),
                opacity=0.7,
                hovertemplate="Date: %{x}<br>Signal: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Add horizontal line at y=0 for signal plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f"{token} Portfolio Analysis",
        showlegend=True,
        height=800,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99),
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Signal", row=2, col=1)

    return fig




# %%


# Add new class to manage indicators efficiently
class IndicatorManager:
    def __init__(self, data_file="saved_data/token_data.csv"):
        self.data_file = data_file
        self.main_df = self._load_data()

    def _load_data(self):
        """Load existing data or create new DataFrame"""
        return pd.read_csv(self.data_file, index_col=0, parse_dates=True)

    def save_data(self):
        """Save data back to file"""
        self.main_df.to_csv(self.data_file)

    def _get_token_columns(self, token):
        """Get indicator column names for a token"""
        return [f"{token}_{suffix}" for suffix in self.indicator_suffixes]

    def update_expanding_indicators(self, df):
        """Update price scaling using expanding window to avoid forward bias"""
        tokens = [col.split("_")[0] for col in df.columns if col.endswith("_price")]

        for token in tokens:
            price_col = f"{token}_price"
            if price_col not in df.columns:
                continue

            # Get complete price history
            price = df[price_col]

            # Use expanding window for min/max to avoid forward bias
            expanding_max = price.expanding().max()
            expanding_min = price.expanding().min()

            # Scale prices between 0-100 using expanding min/max
            price_scaled = np.where(
                expanding_max != expanding_min, (price - expanding_min) * 100 / (expanding_max - expanding_min), 0
            )

            # Update scaled prices
            if f"{token}_price_scaled" not in self.main_df.columns:
                price_col_index = self.main_df.columns.get_loc(f"{token}_price")
                self.main_df.insert(price_col_index + 1, f"{token}_price_scaled", price_scaled)
            else:
                self.main_df[f"{token}_price_scaled"] = price_scaled

    def _get_frequency_split(self, series):
        """
        Detect the transition point from daily to hourly data
        Returns: index position where hourly data begins
        """
        time_diffs = series.index.to_series().diff()
        hourly_mask = time_diffs <= pd.Timedelta(hours=1)

        # Handle first row which will have NaT diff
        hourly_mask.iloc[0] = hourly_mask.iloc[1] if len(hourly_mask) > 1 else True

        if not hourly_mask.any():
            return None  # All daily data

        # Find first hourly timestamp
        first_hourly_idx = hourly_mask.idxmax()
        return first_hourly_idx

    def update_rolling_indicators(self, df):
        """Update rolling indicators from last valid point for each token"""
        tokens = [col.split("_")[0] for col in df.columns if col.endswith("_price")]

        for token in tokens:
            price_col = f"{token}_price"
            if price_col not in df.columns:
                continue

            # Get last valid index for this token's indicators
            smooth_col = f"{token}_smooth"
            last_valid_smooth = (
                self.main_df[smooth_col].last_valid_index() if smooth_col in self.main_df.columns else None
            )

            # For new tokens or complete recalculation
            # Minimum points needed for hourly data (12 daily points * 24 hours)
            min_points_needed = 12 * 24

            if last_valid_smooth is None:
                # No existing smooth data - start from first valid price
                start_idx = df[price_col].first_valid_index()
            else:
                # Get index position of last valid smooth
                last_valid_pos = df.index.get_loc(last_valid_smooth)
                # Calculate start index position, ensuring we don't go before start of data
                start_pos = max(0, last_valid_pos - min_points_needed)
                start_idx = df.index[start_pos]

            # Get price series from start point
            price_series = df.loc[start_idx:, price_col]
            # print(token, len(price_series), price_series.head(10), price_series.tail(10))

            freq_split = self._get_frequency_split(price_series)

            if freq_split is None and len(price_series) > 12:
                # All daily data
                window = 12  # daily window
                smooth_prices = self._calculate_smooth_prices(price_series, window)
                peaks = self._detect_peaks(smooth_prices, price_series, window)
            else:
                # Split processing for daily and hourly data
                daily_data = price_series[:freq_split]
                hourly_data = price_series[freq_split:]

                if len(daily_data) < 12 and len(hourly_data) < 12 * 24:
                    continue

                # Process daily data
                if not daily_data.empty:
                    daily_window = 12
                    daily_smooth = self._calculate_smooth_prices(daily_data, daily_window)
                    daily_peaks = self._detect_peaks(daily_smooth, daily_data, daily_window)
                else:
                    daily_smooth = pd.Series()
                    daily_peaks = pd.Series()

                # Process hourly data
                if not hourly_data.empty:
                    hourly_window = daily_window * 24
                    hourly_smooth = self._calculate_smooth_prices(hourly_data, hourly_window)
                    hourly_peaks = self._detect_peaks(hourly_smooth, hourly_data, hourly_window)
                else:
                    hourly_smooth = pd.Series()
                    hourly_peaks = pd.Series()

                # Combine results, ensuring no duplicate indices
                smooth_prices = pd.concat([daily_smooth, hourly_smooth]).sort_index()
                smooth_prices = smooth_prices[~smooth_prices.index.duplicated(keep="last")]

                peaks = pd.concat([daily_peaks, hourly_peaks]).sort_index()
                peaks = peaks[~peaks.index.duplicated(keep="last")]

            # Update indicators, ensuring indices align
            self.main_df.loc[smooth_prices.index, f"{token}_smooth"] = smooth_prices
            self.main_df.loc[peaks.index, f"{token}_peaks"] = peaks

    def update_portfolio_signals(self, df):
        """Update portfolio signals efficiently"""
        tokens = [col.split("_")[0] for col in df.columns if col.endswith("_price")]

        for token in tokens:
            price_col = f"{token}_price"
            state_col = f"{token}_state"
            signal_col = f"{token}_portfolio_signal"

            # Get last valid signal index
            last_valid = self.main_df[state_col].last_valid_index() if state_col in self.main_df.columns else None

            # For new tokens or complete recalculation
            if last_valid is None:
                start_idx = df[price_col].first_valid_index()
                initial_state = None
            else:
                start_idx = max(df[price_col].first_valid_index(), last_valid - pd.Timedelta(days=1))
                try:
                    state_value = self.main_df.loc[last_valid, state_col]
                    if state_value is None:
                        initial_state = None
                    else:
                        initial_state = eval(state_value)
                        if not isinstance(initial_state, dict):
                            initial_state = None
                except:
                    initial_state = None

            # Calculate signals from start_idx
            signals = calculate_portfolio_signals_for_token(
                self.main_df.loc[start_idx:], self.main_df, token, initial_state=initial_state, send_notifications=True
            )

            # Update signals and state separately with correct dtypes
            # Update portfolio signal (float)
            if signal_col in signals.columns:
                self.main_df.loc[start_idx:, signal_col] = signals[signal_col].astype(float)

            # Update state (object)
            if state_col in signals.columns:
                # Ensure column is object dtype before assignment
                if state_col not in self.main_df.columns:
                    self.main_df[state_col] = pd.Series(dtype="object")
                elif self.main_df[state_col].dtype != "object":
                    self.main_df[state_col] = self.main_df[state_col].astype("object")

                self.main_df.loc[start_idx:, state_col] = signals[state_col]

    def _calculate_smooth_prices(self, series, window, polyorder=2):
        """
        Calculate smoothed prices using Savitzky-Golay filter
        Uses same approach as original implementation
        """

        values = series.values
        result = np.full_like(values, np.nan)

        # First few points just use original values
        min_points = polyorder + 2  # Minimum points needed for smoothing
        result[: polyorder + 2] = values[: polyorder + 2]

        # For each point, fit polynomial to previous window points
        for i in range(min_points, len(values)):
            # Get window of past data
            start_idx = max(0, i - window + 1)
            window_data = values[start_idx : i + 1]  # Include current point

            if len(window_data) > min_points:
                try:
                    # Fit polynomial to past data only
                    x = np.arange(len(window_data))
                    coeffs = np.polyfit(x, window_data, 2)  # quadratic fit

                    # Use polynomial value at current point
                    result[i] = np.polyval(coeffs, len(window_data) - 1)
                except:
                    result[i] = values[i]
            else:
                result[i] = values[i]

        return pd.Series(result, index=series.index)

    def _detect_peaks(self, smooth_prices, raw_prices, window):
        """
        Detect peaks and troughs using find_peaks with minimal lookahead
        """
        signals = pd.Series(0, index=smooth_prices.index)
        prices = smooth_prices.values

        # Parameters for find_peaks
        lookahead = 2  # Maximum forward-looking window
        distance = 3  # Minimum samples between peaks
        # prominence = 0.0005  # Minimum prominence relative to neighbors

        for i in range(window, len(prices) - lookahead):
            # Get local window of prices centered on current point
            start_idx = max(0, i - window // 2)
            end_idx = min(len(prices), i + window // 2 + 1)
            local_window = prices[start_idx:end_idx]

            # Current point's position in local window
            center_idx = i - start_idx

            if window >= 288:  # For hourly data
                distance = 6

            # Find peaks in local window
            peak_indices, _ = find_peaks(
                local_window,
                distance=distance,
            )

            # Find troughs in inverted window
            trough_indices, _ = find_peaks(-local_window, distance=distance)

            # Check if center point is a peak or trough
            if len(peak_indices) > 0 and center_idx in peak_indices:
                # Confirm peak with slope check on local window
                if (
                    center_idx + lookahead < len(local_window)
                    and all(local_window[center_idx] > local_window[center_idx - a] for a in range(1, lookahead + 1))
                    and all(local_window[center_idx] > local_window[center_idx + a] for a in range(1, lookahead + 1))
                ):
                    signals.iloc[i] = 1

            elif len(trough_indices) > 0 and center_idx in trough_indices:
                # Confirm trough with slope check on local window
                if (
                    center_idx + lookahead < len(local_window)
                    and all(local_window[center_idx] < local_window[center_idx - a] for a in range(1, lookahead + 1))
                    and all(local_window[center_idx] < local_window[center_idx + a] for a in range(1, lookahead + 1))
                ):
                    signals.iloc[i] = -1

        return signals

    def process_new_data(self, df):
        """Process new data efficiently"""
        logger = loggers["data"]

        start = time.time()
        self.update_expanding_indicators(df)
        logger.info(f"Expanding indicators updated in {time.time() - start:.2f}s")

        start = time.time()
        self.update_rolling_indicators(df)
        logger.info(f"Rolling indicators updated in {time.time() - start:.2f}s")

        start = time.time()
        self.update_portfolio_signals(df)
        logger.info(f"Portfolio signals updated in {time.time() - start:.2f}s")

        self.save_data()
        logger.info("Data processing complete and saved")


# %%
# file_path = "saved_data/token_data.csv"
# data_manager = CryptoDataManager(data_file_path=file_path)

# data_manager.update_multiple_tokens(TRACKED_TOKENS.keys())


def main(file_path="saved_data/token_data.csv"):
    logger = loggers["system"]

    try:
        logger.info("Starting crypto signals processing")

        data_manager = CryptoDataManager(data_file_path=file_path)
        data_manager.update_multiple_tokens(TRACKED_TOKENS.keys())
        logger.info("Token data updated")

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        validate_token_price_nulls(df)
        logger.info("Token price validation complete")

        notify_price_alerts(df)
        logger.info("Price alerts processed")

        indicator_manager = IndicatorManager(data_file=file_path)
        indicator_manager.process_new_data(df)
        logger.info("Indicator processing complete")

    except Exception as e:
        logger.error("Critical error in main process", exc_info=True)
        raise


main()

# %%
df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

for token in TRACKED_TOKENS.keys():
    plot_portfolio_signals(df, token).show()

# %%
import pandas as pd

# Load token data
df = pd.read_csv("saved_data/saved_data/token_data.csv", index_col=0, parse_dates=True)
df
# %%
# Load token data
df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

# Remove last 20 days of data
cutoff_date = df.index.max() - pd.Timedelta(days=10)
df = df[df.index <= cutoff_date]

# Save truncated data
df.to_csv("saved_data/token_data.csv")

print(f"Saved data up to {cutoff_date}")

# %%


def clear_smooth_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clear all smooth price columns by setting them to None.

    Parameters:
    - df: DataFrame containing token data

    Returns:
    - DataFrame with smooth price columns cleared
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Get list of smooth columns
    smooth_cols = [col for col in df.columns if col.endswith("_smooth")]

    # Set smooth columns to None
    for col in smooth_cols:
        df[col] = None

    return df


# Load data
df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

# Clear smooth price columns
cleaned_df = clear_smooth_price_columns(df)

# Save cleaned data
cleaned_df.to_csv("saved_data/token_data.csv")

print("Smooth price columns cleared successfully")


# %%


def clear_non_price_volume_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clear all columns that are not related to price, price_minmax, or volume.
    Keeps only raw price and volume data.

    Parameters:
    - df: DataFrame containing token data

    Returns:
    - DataFrame with non-price/volume columns cleared
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Get list of columns to keep
    set_to_none_cols = [col for col in df.columns if col.startswith("clearpool")]

    for col in set_to_none_cols:
        df[col] = None

    return df


def main_clear_columns():
    # Load data
    df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

    # Clear non-price/volume columns
    cleaned_df = clear_non_price_volume_columns(df)

    dp(cleaned_df.columns)

    # Save cleaned data
    cleaned_df.to_csv("saved_data/token_data.csv")

    print("Data cleaned and saved successfully")


# Run the cleaning process
main_clear_columns()

# %%


def clear_all_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set all state columns of all tokens to None.
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Find all state columns
    state_cols = [col for col in df.columns if col.endswith("_state")]

    # Set state columns to None with object dtype
    for col in state_cols:
        df[col] = None  # Changed from df.loc[df.index[:], col] = None
        df[col] = df[col].astype("object")  # Ensure object dtype

    return df


def main_clear_states():
    # Load data
    df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

    # Clear state columns
    cleaned_df = clear_all_state_columns(df)

    # Save cleaned data
    cleaned_df.to_csv("saved_data/token_data.csv")

    print("State columns cleared successfully")


main_clear_states()


# %%
def update_state_column_names():
    """
    Load state columns, convert ath_count to peak_count in state dictionaries, and save back to file
    """
    # Load data
    df = pd.read_csv("saved_data/token_data.csv", index_col=0, parse_dates=True)

    # Find all state columns
    state_cols = [col for col in df.columns if col.endswith("_state")]

    # Process each state column
    for col in state_cols:
        # Get non-null state values
        mask = df[col].notna()

        # Convert strings to dicts and update key name
        df.loc[mask, col] = (
            df.loc[mask, col]
            .apply(
                lambda x: {
                    "last_ath_price": eval(x)["last_ath_price"],
                    "peak_count": eval(x)["ath_count"] if "ath_count" in eval(x) else eval(x).get("peak_count", 0),
                    "ath_triggered": eval(x)["ath_triggered"],
                }
            )
            .apply(str)
        )

    # Save updated data
    df.to_csv("saved_data/token_data.csv")
    print("State columns updated successfully")


# Run the update
update_state_column_names()


def validate_state_continuity(df, token):
    """Validate that state transitions are continuous and logical"""
    state_col = f"{token}_state"
    states = df[state_col].dropna()

    prev_state = None
    for idx, state_str in states.items():
        current_state = parse_state(state_str)
        if current_state is None:
            continue

        if prev_state:
            # Verify peak count never decreases
            if current_state["peak_count"] < prev_state["peak_count"]:
                print(f"Warning: Peak count decreased at {idx}")

            # Verify ATH price logic
            if current_state["last_ath_price"] == 0 and prev_state["last_ath_price"] > 0:
                print(f"Warning: ATH price reset to 0 at {idx}")

        prev_state = current_state


def validate_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "CG_API_KEY_1", "CG_API_KEY_2", "CG_API_KEY_3"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )


# Call this at startup
validate_env_vars()
