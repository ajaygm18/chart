"""
Data Loader Module for Chart Pattern Detection System
Handles loading and preprocessing of OHLCV stock data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of OHLCV (Open, High, Low, Close, Volume) stock data.
    Supports both CSV files and Yahoo Finance API.
    """
    
    def __init__(self):
        self.data = None
        self.symbol = None
        
    def load_from_yfinance(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading data for {symbol} with period={period}, interval={interval}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.data = data
            self.symbol = symbol
            logger.info(f"Successfully loaded {len(data)} records for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from Yahoo Finance: {e}")
            raise
            
    def load_from_csv(self, file_path: str, date_column: str = 'Date') -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of the date column
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading data from CSV: {file_path}")
            data = pd.read_csv(file_path)
            
            # Set date as index
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.data = data
            logger.info(f"Successfully loaded {len(data)} records from CSV")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            raise
            
    def preprocess_data(self, normalize: bool = True, resample_freq: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess the loaded data with normalization and resampling.
        
        Args:
            normalize: Whether to normalize price data
            resample_freq: Resampling frequency ('1D', '1W', '1M', etc.)
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_yfinance() or load_from_csv() first.")
            
        data = self.data.copy()
        
        # Remove any missing values
        data = data.dropna()
        
        # Resample if requested
        if resample_freq:
            logger.info(f"Resampling data to {resample_freq}")
            data = data.resample(resample_freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Normalize price data if requested
        if normalize:
            logger.info("Normalizing price data")
            data = self._normalize_prices(data)
            
        return data
        
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data."""
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Price ranges
        data['price_range'] = data['high'] - data['low']
        data['body_size'] = abs(data['close'] - data['open'])
        
        # High-Low spread as percentage of close
        data['hl_pct'] = (data['high'] - data['low']) / data['close'] * 100
        
        return data
        
    def _normalize_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize price columns using min-max scaling."""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                data[f'{col}_normalized'] = (data[col] - min_val) / (max_val - min_val)
                
        return data
        
    def get_price_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get price data for a specific date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded.")
            
        data = self.data.copy()
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        return data
        
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded.")
            
        return {
            'symbol': self.symbol,
            'total_records': len(self.data),
            'date_range': {
                'start': str(self.data.index.min().date()),
                'end': str(self.data.index.max().date())
            },
            'price_summary': {
                'min_close': float(self.data['close'].min()),
                'max_close': float(self.data['close'].max()),
                'avg_close': float(self.data['close'].mean()),
                'avg_volume': float(self.data['volume'].mean())
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader()
    
    try:
        # Load Apple stock data
        data = loader.load_from_yfinance("AAPL", period="6mo")
        
        # Preprocess the data
        processed_data = loader.preprocess_data(normalize=True, resample_freq="1D")
        
        # Get summary
        summary = loader.get_data_summary()
        print("Data Summary:")
        print(f"Symbol: {summary['symbol']}")
        print(f"Records: {summary['total_records']}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Price Range: ${summary['price_summary']['min_close']:.2f} - ${summary['price_summary']['max_close']:.2f}")
        
        print(f"\nProcessed data shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
        
    except Exception as e:
        print(f"Error: {e}")