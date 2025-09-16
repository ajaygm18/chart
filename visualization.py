"""
Visualization Module for Chart Pattern Detection System
Handles plotting stock charts with pattern annotations using matplotlib and plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from pattern_detection import DetectedPattern, PatternType, PatternPoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartVisualizer:
    """
    Handles visualization of stock charts with detected pattern annotations.
    Supports both matplotlib and plotly backends.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), style: str = 'default'):
        """
        Initialize the chart visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
        # Color scheme for different patterns
        self.pattern_colors = {
            PatternType.HEAD_AND_SHOULDERS: '#FF6B6B',
            PatternType.INVERSE_HEAD_AND_SHOULDERS: '#4ECDC4',
            PatternType.DOUBLE_TOP: '#FF8E53',
            PatternType.DOUBLE_BOTTOM: '#95E1D3',
            PatternType.CUP_AND_HANDLE: '#F38BA8',
            PatternType.ASCENDING_TRIANGLE: '#A8E6CF',
            PatternType.DESCENDING_TRIANGLE: '#FFB3BA',
            PatternType.SYMMETRICAL_TRIANGLE: '#B5EAEA',
            PatternType.FLAG: '#C7CEEA',
            PatternType.PENNANT: '#FDBCB4',
            # New pattern colors
            PatternType.RISING_WEDGE: '#FFD93D',
            PatternType.FALLING_WEDGE: '#6BCF7F',
            PatternType.RECTANGLE: '#A8DADC',
            PatternType.ROUNDING_BOTTOM: '#F1FAEE',
            PatternType.ROUNDING_TOP: '#E63946',
            PatternType.TRIPLE_TOP: '#457B9D',
            PatternType.TRIPLE_BOTTOM: '#1D3557',
            PatternType.DIAMOND: '#F72585'
        }
        
    def plot_candlestick_chart(self, data: pd.DataFrame, patterns: List[DetectedPattern] = None,
                              title: str = "Stock Chart with Pattern Detection",
                              save_path: Optional[str] = None, backend: str = 'matplotlib') -> None:
        """
        Plot candlestick chart with pattern annotations.
        
        Args:
            data: DataFrame with OHLCV data
            patterns: List of detected patterns to annotate
            title: Chart title
            save_path: Path to save the chart (optional)
            backend: 'matplotlib' or 'plotly'
        """
        if backend == 'plotly':
            self._plot_plotly_candlestick(data, patterns, title, save_path)
        else:
            self._plot_matplotlib_candlestick(data, patterns, title, save_path)
            
    def _plot_matplotlib_candlestick(self, data: pd.DataFrame, patterns: List[DetectedPattern],
                                    title: str, save_path: Optional[str]) -> None:
        """Plot candlestick chart using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot simple line chart instead of candlesticks to avoid timezone issues
        ax1.plot(data.index, data['close'], linewidth=2, color='blue', label='Close Price')
        ax1.fill_between(data.index, data['low'], data['high'], alpha=0.3, color='lightblue', label='High-Low Range')
        
        # Plot volume
        colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' for i in range(len(data))]
        ax2.bar(data.index, data['volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add pattern annotations
        if patterns:
            self._annotate_patterns_matplotlib(ax1, data, patterns)
            
        # Customize chart
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis
        ax1.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        else:
            plt.show()
            
    def _draw_candlesticks(self, ax, dates, opens, highs, lows, closes):
        """Draw candlestick chart on matplotlib axis."""
        for i in range(len(dates)):
            date = dates[i]
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]
            
            # Determine color
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the high-low line
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
            
            # Draw the body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            # Calculate width as a fraction of the time range
            if len(dates) > 1:
                if i == 0:
                    width = (dates[i+1] - dates[i]) * 0.6
                elif i == len(dates) - 1:
                    width = (dates[i] - dates[i-1]) * 0.6
                else:
                    width = min((dates[i+1] - dates[i]), (dates[i] - dates[i-1])) * 0.6
            else:
                width = pd.Timedelta(hours=16)
            
            rect = Rectangle((date - width/2, body_bottom), 
                           width, body_height,
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
    def _annotate_patterns_matplotlib(self, ax, data: pd.DataFrame, patterns: List[DetectedPattern]):
        """Add pattern annotations to matplotlib chart."""
        for i, pattern in enumerate(patterns):
            color = self.pattern_colors.get(pattern.pattern_type, '#666666')
            
            # Get pattern date range
            start_date = pd.to_datetime(pattern.start_date)
            end_date = pd.to_datetime(pattern.end_date)
            
            # Ensure timezone compatibility
            if data.index.tz is not None:
                if start_date.tz is None:
                    start_date = start_date.tz_localize(data.index.tz)
                if end_date.tz is None:
                    end_date = end_date.tz_localize(data.index.tz)
            else:
                if start_date.tz is not None:
                    start_date = start_date.tz_localize(None)
                if end_date.tz is not None:
                    end_date = end_date.tz_localize(None)
            
            # Find price range for the pattern
            pattern_data = data[(data.index >= start_date) & (data.index <= end_date)]
            if pattern_data.empty:
                continue
                
            min_price = pattern_data[['low', 'close']].min().min()
            max_price = pattern_data[['high', 'close']].max().max()
            price_range = max_price - min_price
            
            # Draw pattern boundary as vertical lines
            ax.axvline(x=start_date, color=color, linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(x=end_date, color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            # Fill pattern area
            pattern_close = pattern_data['close'] if not pattern_data.empty else []
            if not pattern_data.empty:
                ax.fill_between(pattern_data.index, pattern_data['low'], pattern_data['high'], 
                               alpha=0.2, color=color)
            
            # Add pattern label
            if not pattern_data.empty:
                label_y = pattern_data['high'].max() + price_range * 0.02
                label_x = start_date + (end_date - start_date) / 2
                ax.annotate(
                    f"{pattern.pattern_type.value.replace('_', ' ').title()}\n(Conf: {pattern.confidence:.2f})",
                    xy=(label_x, label_y),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=9, fontweight='bold', color='white',
                    ha='center'
                )
            
            # Annotate key points
            self._annotate_pattern_points(ax, pattern, color)
            
            # Draw neckline if exists
            if pattern.neckline:
                ax.axhline(y=pattern.neckline, xmin=0, xmax=1, 
                          color=color, linestyle='--', alpha=0.8, linewidth=2,
                          label=f'{pattern.pattern_type.value} Neckline' if i == 0 else "")
                
    def _annotate_pattern_points(self, ax, pattern: DetectedPattern, color: str):
        """Annotate key points of a pattern."""
        for point in pattern.points:
            # Use the actual date from the point instead of calculating
            try:
                point_date = pd.to_datetime(point.date)
                
                # Plot the point
                ax.plot(point_date, point.price, 'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=2)
                
                # Add point label
                ax.annotate(
                    point.type.replace('_', ' ').title(),
                    xy=(point_date, point.price),
                    xytext=(5, 15), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
                )
            except Exception as e:
                # Skip problematic points
                logger.warning(f"Could not annotate point: {e}")
                continue
            
    def _plot_plotly_candlestick(self, data: pd.DataFrame, patterns: List[DetectedPattern],
                                title: str, save_path: Optional[str]) -> None:
        """Plot interactive candlestick chart using plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume chart
        colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add pattern annotations
        if patterns:
            self._annotate_patterns_plotly(fig, data, patterns)
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis2_title='Date',
            yaxis2_title='Volume',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Configure x-axis to remove gaps for weekends and holidays
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                # Common US market holidays (can be customized)
                dict(values=[
                    "2025-01-01",  # New Year's Day
                    "2025-01-20",  # Martin Luther King Jr. Day
                    "2025-02-17",  # Presidents Day
                    "2025-04-18",  # Good Friday
                    "2025-05-26",  # Memorial Day
                    "2025-07-04",  # Independence Day
                    "2025-09-01",  # Labor Day
                    "2025-11-27",  # Thanksgiving
                    "2025-12-25",  # Christmas Day
                ])
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive chart saved to {save_path}")
        else:
            fig.show()
            
    def _annotate_patterns_plotly(self, fig, data: pd.DataFrame, patterns: List[DetectedPattern]):
        """Add pattern annotations to plotly chart."""
        for pattern in patterns:
            color = self.pattern_colors.get(pattern.pattern_type, '#666666')
            
            # Get pattern date range
            start_date = pattern.start_date
            end_date = pattern.end_date
            
            # Convert to pandas datetime for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Ensure timezone compatibility
            if data.index.tz is not None:
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize(data.index.tz)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize(data.index.tz)
            else:
                if start_dt.tz is not None:
                    start_dt = start_dt.tz_localize(None)
                if end_dt.tz is not None:
                    end_dt = end_dt.tz_localize(None)
            
            # Find price range for the pattern
            pattern_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            if pattern_data.empty:
                continue
                
            min_price = pattern_data[['low', 'close']].min().min()
            max_price = pattern_data[['high', 'close']].max().max()
            
            # Add pattern rectangle
            fig.add_shape(
                type="rect",
                x0=start_date, y0=min_price,
                x1=end_date, y1=max_price,
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.15,
                row=1, col=1
            )
            
            # Add pattern annotation
            fig.add_annotation(
                x=start_date,
                y=max_price,
                text=f"{pattern.pattern_type.value.replace('_', ' ').title()}<br>(Conf: {pattern.confidence:.2f})",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                bgcolor=color,
                bordercolor=color,
                font=dict(color="white", size=10),
                row=1, col=1
            )
            
            # Add neckline if exists
            if pattern.neckline:
                fig.add_hline(
                    y=pattern.neckline,
                    line_dash="dash",
                    line_color=color,
                    line_width=2,
                    row=1, col=1
                )
                
    def create_pattern_summary_chart(self, patterns: List[DetectedPattern], 
                                   save_path: Optional[str] = None) -> None:
        """Create a summary chart showing pattern distribution."""
        if not patterns:
            logger.warning("No patterns to summarize")
            return
            
        # Count patterns by type
        pattern_counts = {}
        confidence_by_type = {}
        
        for pattern in patterns:
            pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
            if pattern_name not in pattern_counts:
                pattern_counts[pattern_name] = 0
                confidence_by_type[pattern_name] = []
            pattern_counts[pattern_name] += 1
            confidence_by_type[pattern_name].append(pattern.confidence)
            
        # Calculate average confidence by type
        avg_confidence = {name: np.mean(conf_list) 
                         for name, conf_list in confidence_by_type.items()}
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Pattern count bar chart
        names = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        colors = [self.pattern_colors.get(PatternType(name.lower().replace(' ', '_')), '#666666') 
                 for name in names]
        
        ax1.bar(names, counts, color=colors, alpha=0.7)
        ax1.set_title('Pattern Count Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Patterns')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Average confidence by pattern type
        confidences = list(avg_confidence.values())
        ax2.bar(names, confidences, color=colors, alpha=0.7)
        ax2.set_title('Average Confidence by Pattern Type', fontweight='bold')
        ax2.set_ylabel('Average Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Pattern confidence distribution
        all_confidences = [p.confidence for p in patterns]
        ax3.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Pattern Confidence Distribution', fontweight='bold')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        
        # 4. Pattern timeline
        pattern_dates = [pd.to_datetime(p.start_date) for p in patterns]
        pattern_types = [p.pattern_type.value.replace('_', ' ').title() for p in patterns]
        
        # Create a timeline scatter plot
        for i, (date, pattern_type) in enumerate(zip(pattern_dates, pattern_types)):
            color = self.pattern_colors.get(PatternType(pattern_type.lower().replace(' ', '_')), '#666666')
            ax4.scatter(date, i, color=color, s=100, alpha=0.7)
            
        ax4.set_title('Pattern Detection Timeline', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Pattern Index')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary chart saved to {save_path}")
        else:
            plt.show()
            
    def create_pattern_detail_chart(self, pattern: DetectedPattern, data: pd.DataFrame,
                                  save_path: Optional[str] = None) -> None:
        """Create a detailed chart for a specific pattern."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get pattern data
        start_date = pd.to_datetime(pattern.start_date)
        end_date = pd.to_datetime(pattern.end_date)
        
        # Ensure timezone compatibility
        if data.index.tz is not None:
            if start_date.tz is None:
                start_date = start_date.tz_localize(data.index.tz)
            if end_date.tz is None:
                end_date = end_date.tz_localize(data.index.tz)
        else:
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if end_date.tz is not None:
                end_date = end_date.tz_localize(None)
        
        # Extend the view a bit before and after the pattern
        buffer_days = (end_date - start_date).days // 4
        view_start = start_date - pd.Timedelta(days=buffer_days)
        view_end = end_date + pd.Timedelta(days=buffer_days)
        
        pattern_data = data[(data.index >= view_start) & (data.index <= view_end)]
        
        if pattern_data.empty:
            logger.warning("No data available for pattern detail chart")
            return
            
        # Plot price line
        ax.plot(pattern_data.index, pattern_data['close'], linewidth=2, color='blue', alpha=0.8)
        
        # Highlight pattern area
        pattern_area = data[(data.index >= start_date) & (data.index <= end_date)]
        ax.fill_between(pattern_area.index, pattern_area['low'], pattern_area['high'], 
                       alpha=0.3, color=self.pattern_colors.get(pattern.pattern_type, '#666666'))
        
        # Mark key points
        for point in pattern.points:
            try:
                point_date = pd.to_datetime(point.date)
                # Ensure timezone compatibility
                if data.index.tz is not None and point_date.tz is None:
                    point_date = point_date.tz_localize(data.index.tz)
                elif data.index.tz is None and point_date.tz is not None:
                    point_date = point_date.tz_localize(None)
                    
                if point_date in pattern_data.index:
                    ax.plot(point_date, point.price, 'o', markersize=10, 
                           color=self.pattern_colors.get(pattern.pattern_type, '#666666'),
                           markeredgecolor='white', markeredgewidth=2)
                    
                    # Add label
                    ax.annotate(point.type.replace('_', ' ').title(),
                               xy=(point_date, point.price),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                               fontsize=10, fontweight='bold')
            except Exception as e:
                logger.warning(f"Could not mark point {point.type}: {e}")
                continue
        
        # Add neckline if exists
        if pattern.neckline:
            ax.axhline(y=pattern.neckline, color='red', linestyle='--', linewidth=2,
                      label='Neckline', alpha=0.8)
            
        # Add target price if exists
        if pattern.target_price:
            ax.axhline(y=pattern.target_price, color='green', linestyle=':', linewidth=2,
                      label='Target Price', alpha=0.8)
        
        # Customize chart
        title = f"{pattern.pattern_type.value.replace('_', ' ').title()} Pattern Detail"
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add pattern info text box
        info_text = f"""Pattern: {pattern.pattern_type.value.replace('_', ' ').title()}
Confidence: {pattern.confidence:.2f}
Duration: {pattern.start_date} to {pattern.end_date}"""
        
        if pattern.target_price:
            info_text += f"\nTarget Price: ${pattern.target_price:.2f}"
        if pattern.neckline:
            info_text += f"\nNeckline: ${pattern.neckline:.2f}"
            
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pattern detail chart saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    print("Chart Visualization Module - Ready for integration")
    
    # Example of creating a visualizer
    visualizer = ChartVisualizer(figsize=(15, 10))
    print(f"Visualizer initialized with figsize={visualizer.figsize}")
    print(f"Available pattern colors: {len(visualizer.pattern_colors)} colors defined")