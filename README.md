# Chart Pattern Detection System

A comprehensive Python system for detecting classical chart patterns in stock market data using peak/valley analysis and visualization.

## Features

- **Data Loading**: Support for Yahoo Finance API and CSV files
- **Pattern Detection**: Detects 10+ classical chart patterns including:
  - Head & Shoulders (Regular and Inverse)
  - Double Top/Bottom
  - Cup & Handle
  - Triangle patterns (Ascending, Descending, Symmetrical)
  - Flag and Pennant patterns
- **Visualization**: Interactive and static charts with pattern annotations
- **Export**: JSON results and multiple chart formats
- **Modular Design**: Easy to extend and customize

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chart
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Demo Analysis
Run the demo with Apple stock:
```bash
python main.py
```

### Analyze a Specific Stock
```bash
python main.py --symbol TSLA --period 1y --interval 1d
```

### Analyze CSV Data
```bash
python main.py --csv data/stock_data.csv
```

## Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

**Options:**
- `--symbol, -s`: Stock symbol to analyze (default: AAPL)
- `--period, -p`: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `--interval, -i`: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
- `--csv`: Path to CSV file (alternative to symbol)
- `--output, -o`: Output directory (default: output)
- `--no-patterns`: Skip pattern detection
- `--no-viz`: Skip visualization creation

### Programmatic Usage

```python
from main import ChartPatternAnalyzer

# Create analyzer
analyzer = ChartPatternAnalyzer(output_dir="my_analysis")

# Analyze a stock symbol
results = analyzer.analyze_symbol("AAPL", period="6mo")

# Print summary
analyzer.print_analysis_summary()

# Get detailed statistics
stats = analyzer.get_pattern_statistics()
```

### Individual Module Usage

#### Data Loading
```python
from data_loader import DataLoader

loader = DataLoader()

# Load from Yahoo Finance
data = loader.load_from_yfinance("AAPL", period="1y")

# Load from CSV
data = loader.load_from_csv("data.csv")

# Preprocess data
processed = loader.preprocess_data(normalize=True, resample_freq="1D")
```

#### Pattern Detection
```python
from pattern_detection import PatternDetector

detector = PatternDetector(min_pattern_length=20, peak_prominence=0.02)
patterns = detector.detect_patterns(data)

# Get summary
summary = detector.get_pattern_summary()
```

#### Visualization
```python
from visualization import ChartVisualizer

visualizer = ChartVisualizer()

# Create candlestick chart with patterns
visualizer.plot_candlestick_chart(
    data, patterns, 
    title="My Analysis",
    save_path="chart.png",
    backend="matplotlib"  # or "plotly"
)

# Create pattern summary
visualizer.create_pattern_summary_chart(patterns, "summary.png")
```

## Supported Chart Patterns

### 1. Head and Shoulders
- **Regular**: Bearish reversal pattern
- **Inverse**: Bullish reversal pattern
- Detection based on three peaks with middle peak being highest

### 2. Double Top/Bottom
- **Double Top**: Bearish reversal with two peaks at similar levels
- **Double Bottom**: Bullish reversal with two valleys at similar levels

### 3. Cup and Handle
- Bullish continuation pattern
- U-shaped cup followed by smaller handle formation

### 4. Triangle Patterns
- **Ascending**: Bullish continuation with rising support line
- **Descending**: Bearish continuation with falling resistance line
- **Symmetrical**: Neutral pattern with converging trend lines

### 5. Flag and Pennant
- **Flag**: Short-term continuation with parallel trend lines
- **Pennant**: Short-term continuation with converging trend lines

## Output Files

The system generates several output files:

1. **Pattern Chart** (`{symbol}_pattern_chart.png`): Static candlestick chart with pattern annotations
2. **Interactive Chart** (`{symbol}_interactive_chart.html`): Interactive Plotly chart
3. **Pattern Summary** (`{symbol}_pattern_summary.png`): Statistical summary charts
4. **Pattern Details** (`{symbol}_pattern_{n}_{type}.png`): Individual pattern detail charts
5. **JSON Results** (`{symbol}_analysis_results.json`): Complete analysis data in JSON format

## JSON Output Format

```json
{
  "analysis_info": {
    "symbol": "AAPL",
    "analysis_date": "2024-01-01T12:00:00",
    "data_period": {
      "start": "2023-01-01",
      "end": "2023-12-31",
      "total_days": 365
    }
  },
  "data_summary": {
    "symbol": "AAPL",
    "total_records": 252,
    "date_range": {...},
    "price_summary": {...}
  },
  "pattern_summary": {
    "total_patterns": 5,
    "patterns_by_type": {...},
    "average_confidence": 0.75
  },
  "detected_patterns": [
    {
      "id": 1,
      "type": "head_and_shoulders",
      "confidence": 0.82,
      "start_date": "2023-03-15",
      "end_date": "2023-04-20",
      "neckline": 150.25,
      "target_price": 145.50,
      "key_points": [...]
    }
  ]
}
```

## Configuration

### Pattern Detector Parameters
- `min_pattern_length`: Minimum data points for pattern (default: 20)
- `peak_prominence`: Minimum peak prominence as % of price range (default: 0.02)

### Visualization Options
- Multiple backends: matplotlib (static) and plotly (interactive)
- Customizable color schemes for different patterns
- Configurable chart sizes and styles

## Data Requirements

### Yahoo Finance (Automatic)
- Any valid stock symbol (e.g., AAPL, MSFT, TSLA)
- Automatic OHLCV data retrieval
- Various time periods and intervals supported

### CSV Format
Required columns (case-insensitive):
- Date (or custom date column name)
- Open
- High  
- Low
- Close
- Volume

Example CSV:
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,150.00,152.50,149.00,151.25,1000000
2023-01-02,151.25,153.00,150.50,152.75,1200000
```

## Algorithm Details

### Peak/Valley Detection
- Uses `scipy.signal.find_peaks` for identifying local extrema
- Configurable prominence thresholds
- Distance-based filtering to avoid noise

### Pattern Recognition
Each pattern has specific geometric criteria:
- **Head & Shoulders**: Three peaks with middle highest, symmetric shoulders
- **Double Top/Bottom**: Two peaks/valleys at similar levels
- **Triangles**: Trend line analysis with slope calculations
- **Cup & Handle**: U-shaped formation with small pullback

### Confidence Scoring
Patterns are scored based on:
- Geometric accuracy
- Pattern duration
- Symmetry (where applicable)
- Volume confirmation (when available)

## Examples

### Basic Analysis
```python
from main import ChartPatternAnalyzer

analyzer = ChartPatternAnalyzer()
results = analyzer.analyze_symbol("AAPL", period="6mo")
analyzer.print_analysis_summary()
```

### Custom Configuration
```python
from pattern_detection import PatternDetector
from visualization import ChartVisualizer

# Custom detector settings
detector = PatternDetector(
    min_pattern_length=30,
    peak_prominence=0.03
)

# Custom visualization
visualizer = ChartVisualizer(
    figsize=(20, 12),
    style='seaborn'
)
```

### Batch Analysis
```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
analyzer = ChartPatternAnalyzer()

for symbol in symbols:
    print(f"Analyzing {symbol}...")
    results = analyzer.analyze_symbol(symbol, period="1y")
    print(f"Found {results['patterns_detected']} patterns")
```

## Troubleshooting

### Common Issues

1. **No data found for symbol**
   - Check if symbol is valid
   - Try different time period
   - Verify internet connection

2. **No patterns detected**
   - Reduce `peak_prominence` parameter
   - Increase analysis period
   - Check if data has sufficient volatility

3. **Memory issues with large datasets**
   - Use shorter time periods
   - Resample to daily data
   - Reduce chart resolution

### Performance Tips

- Use daily intervals for long-term analysis
- Enable only necessary visualizations for batch processing
- Cache data for repeated analysis

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.9.0
- scikit-learn >= 1.1.0
- plotly >= 5.10.0
- yfinance >= 0.1.87
- mplfinance >= 0.12.9
- ta >= 0.10.2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new patterns or features
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built using financial data from Yahoo Finance
- Pattern detection algorithms based on classical technical analysis
- Visualization powered by matplotlib and plotly