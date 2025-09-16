"""
Main Module for Chart Pattern Detection System
Integrates data loading, pattern detection, and visualization for complete functionality.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import argparse

from data_loader import DataLoader
from pattern_detection import PatternDetector, DetectedPattern, PatternType
from visualization import ChartVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChartPatternAnalyzer:
    """
    Main class that orchestrates the entire chart pattern detection workflow.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save output files
        """
        self.data_loader = DataLoader()
        self.pattern_detector = PatternDetector()
        self.visualizer = ChartVisualizer()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.data = None
        self.patterns = []
        self.symbol = None
        
    def analyze_symbol(self, symbol: str, period: str = "1y", interval: str = "1d",
                      detect_patterns: bool = True, create_visualizations: bool = True) -> Dict:
        """
        Complete analysis workflow for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period
            interval: Data interval
            detect_patterns: Whether to detect patterns
            create_visualizations: Whether to create charts
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting analysis for {symbol}")
        self.symbol = symbol
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading stock data...")
            self.data = self.data_loader.load_from_yfinance(symbol, period, interval)
            
            # Step 2: Preprocess data
            logger.info("Step 2: Preprocessing data...")
            processed_data = self.data_loader.preprocess_data(normalize=False, resample_freq=None)
            
            # Step 3: Detect patterns
            results = {"symbol": symbol, "analysis_date": datetime.now().isoformat()}
            
            if detect_patterns:
                logger.info("Step 3: Detecting chart patterns...")
                self.patterns = self.pattern_detector.detect_patterns(processed_data)
                results["patterns_detected"] = len(self.patterns)
                results["pattern_summary"] = self.pattern_detector.get_pattern_summary()
            else:
                self.patterns = []
                results["patterns_detected"] = 0
                
            # Step 4: Create visualizations
            if create_visualizations:
                logger.info("Step 4: Creating visualizations...")
                self._create_all_visualizations()
                results["visualizations_created"] = True
            else:
                results["visualizations_created"] = False
                
            # Step 5: Export results
            logger.info("Step 5: Exporting results...")
            json_path = self._export_results_to_json()
            results["json_export_path"] = json_path
            
            logger.info(f"Analysis completed successfully for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
            
    def analyze_csv_file(self, csv_path: str, date_column: str = "Date",
                        detect_patterns: bool = True, create_visualizations: bool = True) -> Dict:
        """
        Analyze data from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            date_column: Name of date column
            detect_patterns: Whether to detect patterns
            create_visualizations: Whether to create charts
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting analysis for CSV file: {csv_path}")
        self.symbol = os.path.basename(csv_path).replace('.csv', '')
        
        try:
            # Load data from CSV
            logger.info("Loading data from CSV...")
            self.data = self.data_loader.load_from_csv(csv_path, date_column)
            
            # Continue with same workflow as analyze_symbol
            processed_data = self.data_loader.preprocess_data(normalize=False)
            
            results = {"source": csv_path, "analysis_date": datetime.now().isoformat()}
            
            if detect_patterns:
                logger.info("Detecting chart patterns...")
                self.patterns = self.pattern_detector.detect_patterns(processed_data)
                results["patterns_detected"] = len(self.patterns)
                results["pattern_summary"] = self.pattern_detector.get_pattern_summary()
            else:
                self.patterns = []
                results["patterns_detected"] = 0
                
            if create_visualizations:
                logger.info("Creating visualizations...")
                self._create_all_visualizations()
                results["visualizations_created"] = True
            else:
                results["visualizations_created"] = False
                
            json_path = self._export_results_to_json()
            results["json_export_path"] = json_path
            
            logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during CSV analysis: {e}")
            raise
            
    def _create_all_visualizations(self):
        """Create all visualization outputs."""
        if self.data is None:
            logger.warning("No data available for visualization")
            return
            
        symbol_safe = self.symbol.replace('/', '_').replace('\\', '_')
        
        # 1. Main candlestick chart with patterns (matplotlib)
        chart_path = os.path.join(self.output_dir, f"{symbol_safe}_pattern_chart.png")
        self.visualizer.plot_candlestick_chart(
            self.data, self.patterns,
            title=f"{self.symbol} - Chart Pattern Analysis",
            save_path=chart_path,
            backend='matplotlib'
        )
        
        # 2. Interactive chart (plotly)
        interactive_path = os.path.join(self.output_dir, f"{symbol_safe}_interactive_chart.html")
        self.visualizer.plot_candlestick_chart(
            self.data, self.patterns,
            title=f"{self.symbol} - Interactive Chart Pattern Analysis",
            save_path=interactive_path,
            backend='plotly'
        )
        
        # 3. Pattern summary charts
        if self.patterns:
            summary_path = os.path.join(self.output_dir, f"{symbol_safe}_pattern_summary.png")
            self.visualizer.create_pattern_summary_chart(self.patterns, summary_path)
            
            # 4. Individual pattern detail charts
            for i, pattern in enumerate(self.patterns):
                detail_path = os.path.join(
                    self.output_dir, 
                    f"{symbol_safe}_pattern_{i+1}_{pattern.pattern_type.value}.png"
                )
                self.visualizer.create_pattern_detail_chart(pattern, self.data, detail_path)
                
    def _export_results_to_json(self) -> str:
        """Export analysis results to JSON file."""
        if self.data is None:
            logger.warning("No data available for export")
            return ""
            
        # Prepare data for JSON export
        results = {
            "analysis_info": {
                "symbol": self.symbol,
                "analysis_date": datetime.now().isoformat(),
                "data_period": {
                    "start": str(self.data.index.min().date()),
                    "end": str(self.data.index.max().date()),
                    "total_days": len(self.data)
                }
            },
            "data_summary": self.data_loader.get_data_summary(),
            "pattern_summary": self.pattern_detector.get_pattern_summary(),
            "detected_patterns": []
        }
        
        # Add detailed pattern information
        for i, pattern in enumerate(self.patterns):
            pattern_dict = {
                "id": i + 1,
                "type": pattern.pattern_type.value,
                "confidence": pattern.confidence,
                "start_date": pattern.start_date,
                "end_date": pattern.end_date,
                "description": pattern.description,
                "neckline": pattern.neckline,
                "target_price": pattern.target_price,
                "stop_loss": pattern.stop_loss,
                "key_points": [
                    {
                        "index": point.index,
                        "price": point.price,
                        "date": point.date,
                        "type": point.type
                    }
                    for point in pattern.points
                ]
            }
            results["detected_patterns"].append(pattern_dict)
            
        # Save to JSON file
        symbol_safe = self.symbol.replace('/', '_').replace('\\', '_')
        json_path = os.path.join(self.output_dir, f"{symbol_safe}_analysis_results.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Results exported to {json_path}")
        return json_path
        
    def get_pattern_statistics(self) -> Dict:
        """Get detailed statistics about detected patterns."""
        if not self.patterns:
            return {"message": "No patterns detected"}
            
        stats = {
            "total_patterns": len(self.patterns),
            "pattern_types": {},
            "confidence_stats": {
                "average": np.mean([p.confidence for p in self.patterns]),
                "median": np.median([p.confidence for p in self.patterns]),
                "min": min([p.confidence for p in self.patterns]),
                "max": max([p.confidence for p in self.patterns])
            },
            "high_confidence_patterns": len([p for p in self.patterns if p.confidence > 0.8]),
            "medium_confidence_patterns": len([p for p in self.patterns if 0.6 < p.confidence <= 0.8]),
            "low_confidence_patterns": len([p for p in self.patterns if p.confidence <= 0.6])
        }
        
        # Count by pattern type
        for pattern in self.patterns:
            pattern_name = pattern.pattern_type.value
            if pattern_name not in stats["pattern_types"]:
                stats["pattern_types"][pattern_name] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "patterns": []
                }
            stats["pattern_types"][pattern_name]["count"] += 1
            stats["pattern_types"][pattern_name]["patterns"].append({
                "start_date": pattern.start_date,
                "end_date": pattern.end_date,
                "confidence": pattern.confidence
            })
            
        # Calculate average confidence by type
        for pattern_type in stats["pattern_types"]:
            confidences = [p["confidence"] for p in stats["pattern_types"][pattern_type]["patterns"]]
            stats["pattern_types"][pattern_type]["avg_confidence"] = np.mean(confidences)
            
        return stats
        
    def print_analysis_summary(self):
        """Print a summary of the analysis results."""
        if self.data is None:
            print("No data loaded for analysis.")
            return
            
        print(f"\n{'='*60}")
        print(f"CHART PATTERN ANALYSIS SUMMARY - {self.symbol}")
        print(f"{'='*60}")
        
        # Data summary
        summary = self.data_loader.get_data_summary()
        print(f"\nDATA SUMMARY:")
        print(f"  Symbol: {summary['symbol']}")
        print(f"  Records: {summary['total_records']}")
        print(f"  Period: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Price Range: ${summary['price_summary']['min_close']:.2f} - ${summary['price_summary']['max_close']:.2f}")
        print(f"  Average Volume: {summary['price_summary']['avg_volume']:,.0f}")
        
        # Pattern summary
        if self.patterns:
            print(f"\nPATTERN DETECTION RESULTS:")
            print(f"  Total Patterns Found: {len(self.patterns)}")
            
            pattern_stats = self.get_pattern_statistics()
            print(f"  Average Confidence: {pattern_stats['confidence_stats']['average']:.3f}")
            print(f"  High Confidence (>0.8): {pattern_stats['high_confidence_patterns']}")
            print(f"  Medium Confidence (0.6-0.8): {pattern_stats['medium_confidence_patterns']}")
            print(f"  Low Confidence (≤0.6): {pattern_stats['low_confidence_patterns']}")
            
            print(f"\n  PATTERNS BY TYPE:")
            for pattern_type, info in pattern_stats["pattern_types"].items():
                print(f"    {pattern_type.replace('_', ' ').title()}: {info['count']} "
                      f"(Avg Conf: {info['avg_confidence']:.3f})")
                      
            print(f"\n  DETECTED PATTERNS:")
            for i, pattern in enumerate(self.patterns, 1):
                print(f"    {i}. {pattern.pattern_type.value.replace('_', ' ').title()}")
                print(f"       Period: {pattern.start_date} to {pattern.end_date}")
                print(f"       Confidence: {pattern.confidence:.3f}")
                if pattern.target_price:
                    print(f"       Target Price: ${pattern.target_price:.2f}")
                if pattern.neckline:
                    print(f"       Neckline: ${pattern.neckline:.2f}")
                print()
        else:
            print(f"\nPATTERN DETECTION RESULTS:")
            print(f"  No patterns detected in the analyzed period.")
            
        print(f"\nOutput files saved to: {self.output_dir}/")
        print(f"{'='*60}\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Chart Pattern Detection System")
    parser.add_argument("--symbol", "-s", type=str, default="AAPL", 
                       help="Stock symbol to analyze (default: AAPL)")
    parser.add_argument("--period", "-p", type=str, default="1y",
                       help="Data period (default: 1y)")
    parser.add_argument("--interval", "-i", type=str, default="1d",
                       help="Data interval (default: 1d)")
    parser.add_argument("--csv", type=str, help="CSV file path (alternative to symbol)")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Output directory (default: output)")
    parser.add_argument("--no-patterns", action="store_true",
                       help="Skip pattern detection")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization creation")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ChartPatternAnalyzer(output_dir=args.output)
    
    try:
        if args.csv:
            # Analyze CSV file
            results = analyzer.analyze_csv_file(
                args.csv,
                detect_patterns=not args.no_patterns,
                create_visualizations=not args.no_viz
            )
        else:
            # Analyze symbol from Yahoo Finance
            results = analyzer.analyze_symbol(
                args.symbol,
                period=args.period,
                interval=args.interval,
                detect_patterns=not args.no_patterns,
                create_visualizations=not args.no_viz
            )
            
        # Print summary
        analyzer.print_analysis_summary()
        
        print("Analysis completed successfully!")
        print(f"Check the '{args.output}' directory for generated files.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")


# Example usage and demo
def demo_analysis():
    """Run a demo analysis with Apple stock."""
    print("Running Chart Pattern Detection Demo...")
    print("Analyzing Apple (AAPL) stock for the past 6 months...")
    
    analyzer = ChartPatternAnalyzer(output_dir="demo_output")
    
    try:
        # Analyze Apple stock
        results = analyzer.analyze_symbol("AAPL", period="6mo", interval="1d")
        
        # Print results
        analyzer.print_analysis_summary()
        
        # Show some additional statistics
        stats = analyzer.get_pattern_statistics()
        
        if stats.get("total_patterns", 0) > 0:
            print("DETAILED PATTERN STATISTICS:")
            print(json.dumps(stats, indent=2, default=str))
        
        print("\nDemo completed! Check 'demo_output' directory for generated files.")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo_analysis()
    else:
        # Arguments provided, run CLI
        main()