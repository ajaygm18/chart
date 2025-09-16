"""
Pattern Detection Module for Chart Pattern Detection System
Implements algorithms to detect classical chart patterns using peak/valley analysis.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Enumeration of supported chart patterns."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    CUP_AND_HANDLE = "cup_and_handle"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"


@dataclass
class PatternPoint:
    """Represents a significant point in a chart pattern."""
    index: int
    price: float
    date: str
    type: str  # 'peak', 'valley', 'neckline', etc.


@dataclass
class DetectedPattern:
    """Represents a detected chart pattern."""
    pattern_type: PatternType
    confidence: float
    start_date: str
    end_date: str
    points: List[PatternPoint]
    neckline: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""


class PatternDetector:
    """
    Main class for detecting chart patterns in OHLCV data.
    Uses peak/valley detection and geometric analysis.
    """
    
    def __init__(self, min_pattern_length: int = 20, peak_prominence: float = 0.02):
        """
        Initialize the pattern detector.
        
        Args:
            min_pattern_length: Minimum number of data points for a pattern
            peak_prominence: Minimum prominence for peak detection (as percentage of price range)
        """
        self.min_pattern_length = min_pattern_length
        self.peak_prominence = peak_prominence
        self.detected_patterns = []
        
    def detect_patterns(self, data: pd.DataFrame, price_column: str = 'close') -> List[DetectedPattern]:
        """
        Detect all supported patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Column to use for pattern detection
            
        Returns:
            List of detected patterns
        """
        self.detected_patterns = []
        
        if len(data) < self.min_pattern_length:
            logger.warning(f"Data too short for pattern detection. Need at least {self.min_pattern_length} points.")
            return []
            
        logger.info(f"Detecting patterns in {len(data)} data points")
        
        # Find peaks and valleys
        peaks, valleys = self._find_peaks_and_valleys(data, price_column)
        
        if len(peaks) < 2 or len(valleys) < 2:
            logger.warning("Insufficient peaks/valleys for pattern detection")
            return []
            
        # Detect different pattern types
        self._detect_head_and_shoulders(data, peaks, valleys, price_column)
        self._detect_double_top_bottom(data, peaks, valleys, price_column)
        self._detect_triangles(data, peaks, valleys, price_column)
        self._detect_cup_and_handle(data, peaks, valleys, price_column)
        self._detect_flags_and_pennants(data, peaks, valleys, price_column)
        
        logger.info(f"Detected {len(self.detected_patterns)} patterns")
        return self.detected_patterns
        
    def _find_peaks_and_valleys(self, data: pd.DataFrame, price_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Find significant peaks and valleys in the price data."""
        prices = data[price_column].values
        price_range = prices.max() - prices.min()
        prominence = price_range * self.peak_prominence
        
        # Find peaks (local maxima)
        peaks, _ = find_peaks(prices, prominence=prominence, distance=5)
        
        # Find valleys (local minima) by finding peaks in inverted data
        valleys, _ = find_peaks(-prices, prominence=prominence, distance=5)
        
        logger.debug(f"Found {len(peaks)} peaks and {len(valleys)} valleys")
        return peaks, valleys
        
    def _detect_head_and_shoulders(self, data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, price_column: str):
        """Detect Head and Shoulders and Inverse Head and Shoulders patterns."""
        if len(peaks) < 3 or len(valleys) < 2:
            return
            
        prices = data[price_column].values
        dates = data.index
        
        # Check each triplet of peaks for head and shoulders
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Find valleys between peaks
            valleys_between = valleys[(valleys > left_shoulder) & (valleys < right_shoulder)]
            if len(valleys_between) < 2:
                continue
                
            left_valley = valleys_between[0]
            right_valley = valleys_between[-1]
            
            # Check head and shoulders criteria
            if self._is_head_and_shoulders(prices, left_shoulder, head, right_shoulder, left_valley, right_valley):
                pattern = self._create_head_and_shoulders_pattern(
                    data, left_shoulder, head, right_shoulder, left_valley, right_valley, 
                    PatternType.HEAD_AND_SHOULDERS
                )
                self.detected_patterns.append(pattern)
                
        # Check for inverse head and shoulders using valleys
        for i in range(len(valleys) - 2):
            left_shoulder = valleys[i]
            head = valleys[i + 1]
            right_shoulder = valleys[i + 2]
            
            # Find peaks between valleys
            peaks_between = peaks[(peaks > left_shoulder) & (peaks < right_shoulder)]
            if len(peaks_between) < 2:
                continue
                
            left_peak = peaks_between[0]
            right_peak = peaks_between[-1]
            
            # Check inverse head and shoulders criteria
            if self._is_inverse_head_and_shoulders(prices, left_shoulder, head, right_shoulder, left_peak, right_peak):
                pattern = self._create_head_and_shoulders_pattern(
                    data, left_shoulder, head, right_shoulder, left_peak, right_peak,
                    PatternType.INVERSE_HEAD_AND_SHOULDERS
                )
                self.detected_patterns.append(pattern)
                
    def _is_head_and_shoulders(self, prices: np.ndarray, ls: int, h: int, rs: int, lv: int, rv: int) -> bool:
        """Check if points form a valid head and shoulders pattern."""
        # Head should be higher than both shoulders
        if not (prices[h] > prices[ls] and prices[h] > prices[rs]):
            return False
            
        # Shoulders should be roughly at the same level (within 5% tolerance)
        shoulder_tolerance = 0.05
        if abs(prices[ls] - prices[rs]) / prices[ls] > shoulder_tolerance:
            return False
            
        # Valleys should be roughly at the same level (neckline)
        neckline_tolerance = 0.03
        if abs(prices[lv] - prices[rv]) / prices[lv] > neckline_tolerance:
            return False
            
        return True
        
    def _is_inverse_head_and_shoulders(self, prices: np.ndarray, ls: int, h: int, rs: int, lp: int, rp: int) -> bool:
        """Check if points form a valid inverse head and shoulders pattern."""
        # Head should be lower than both shoulders
        if not (prices[h] < prices[ls] and prices[h] < prices[rs]):
            return False
            
        # Shoulders should be roughly at the same level
        shoulder_tolerance = 0.05
        if abs(prices[ls] - prices[rs]) / prices[ls] > shoulder_tolerance:
            return False
            
        # Peaks should be roughly at the same level (neckline)
        neckline_tolerance = 0.03
        if abs(prices[lp] - prices[rp]) / prices[lp] > neckline_tolerance:
            return False
            
        return True
        
    def _create_head_and_shoulders_pattern(self, data: pd.DataFrame, ls: int, h: int, rs: int, 
                                         v1: int, v2: int, pattern_type: PatternType) -> DetectedPattern:
        """Create a head and shoulders pattern object."""
        prices = data[data.columns[3]].values  # close price
        dates = data.index
        
        points = [
            PatternPoint(ls, prices[ls], str(dates[ls].date()), "left_shoulder"),
            PatternPoint(v1, prices[v1], str(dates[v1].date()), "valley"),
            PatternPoint(h, prices[h], str(dates[h].date()), "head"),
            PatternPoint(v2, prices[v2], str(dates[v2].date()), "valley"),
            PatternPoint(rs, prices[rs], str(dates[rs].date()), "right_shoulder")
        ]
        
        # Calculate neckline (average of valleys)
        neckline = (prices[v1] + prices[v2]) / 2
        
        # Calculate target price (height of head above neckline projected below neckline)
        if pattern_type == PatternType.HEAD_AND_SHOULDERS:
            head_height = prices[h] - neckline
            target_price = neckline - head_height
        else:  # Inverse head and shoulders
            head_height = neckline - prices[h]
            target_price = neckline + head_height
            
        # Calculate confidence based on pattern quality
        confidence = self._calculate_pattern_confidence(prices, points)
        
        return DetectedPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            start_date=str(dates[ls].date()),
            end_date=str(dates[rs].date()),
            points=points,
            neckline=neckline,
            target_price=target_price,
            description=f"{pattern_type.value.replace('_', ' ').title()} pattern"
        )
        
    def _detect_double_top_bottom(self, data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, price_column: str):
        """Detect Double Top and Double Bottom patterns."""
        prices = data[price_column].values
        dates = data.index
        
        # Double Top detection
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1, peak2 = peaks[i], peaks[j]
                
                # Check if peaks are at similar levels
                if abs(prices[peak1] - prices[peak2]) / prices[peak1] < 0.03:
                    # Find valley between peaks
                    valleys_between = valleys[(valleys > peak1) & (valleys < peak2)]
                    if len(valleys_between) > 0:
                        valley = valleys_between[np.argmin(prices[valleys_between])]
                        
                        if self._is_double_top(prices, peak1, peak2, valley):
                            pattern = self._create_double_pattern(
                                data, peak1, peak2, valley, PatternType.DOUBLE_TOP
                            )
                            self.detected_patterns.append(pattern)
                            
        # Double Bottom detection
        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                valley1, valley2 = valleys[i], valleys[j]
                
                # Check if valleys are at similar levels
                if abs(prices[valley1] - prices[valley2]) / prices[valley1] < 0.03:
                    # Find peak between valleys
                    peaks_between = peaks[(peaks > valley1) & (peaks < valley2)]
                    if len(peaks_between) > 0:
                        peak = peaks_between[np.argmax(prices[peaks_between])]
                        
                        if self._is_double_bottom(prices, valley1, valley2, peak):
                            pattern = self._create_double_pattern(
                                data, valley1, valley2, peak, PatternType.DOUBLE_BOTTOM
                            )
                            self.detected_patterns.append(pattern)
                            
    def _is_double_top(self, prices: np.ndarray, peak1: int, peak2: int, valley: int) -> bool:
        """Check if points form a valid double top pattern."""
        # Valley should be significantly below peaks
        valley_depth = min(prices[peak1], prices[peak2]) - prices[valley]
        peak_height = (prices[peak1] + prices[peak2]) / 2
        
        return valley_depth / peak_height > 0.03
        
    def _is_double_bottom(self, prices: np.ndarray, valley1: int, valley2: int, peak: int) -> bool:
        """Check if points form a valid double bottom pattern."""
        # Peak should be significantly above valleys
        peak_height = prices[peak] - max(prices[valley1], prices[valley2])
        valley_depth = (prices[valley1] + prices[valley2]) / 2
        
        return peak_height / valley_depth > 0.03
        
    def _create_double_pattern(self, data: pd.DataFrame, point1: int, point2: int, 
                              middle: int, pattern_type: PatternType) -> DetectedPattern:
        """Create a double top/bottom pattern object."""
        prices = data[data.columns[3]].values
        dates = data.index
        
        if pattern_type == PatternType.DOUBLE_TOP:
            points = [
                PatternPoint(point1, prices[point1], str(dates[point1].date()), "first_top"),
                PatternPoint(middle, prices[middle], str(dates[middle].date()), "valley"),
                PatternPoint(point2, prices[point2], str(dates[point2].date()), "second_top")
            ]
            neckline = prices[middle]
            height = (prices[point1] + prices[point2]) / 2 - neckline
            target_price = neckline - height
        else:  # Double Bottom
            points = [
                PatternPoint(point1, prices[point1], str(dates[point1].date()), "first_bottom"),
                PatternPoint(middle, prices[middle], str(dates[middle].date()), "peak"),
                PatternPoint(point2, prices[point2], str(dates[point2].date()), "second_bottom")
            ]
            neckline = prices[middle]
            height = neckline - (prices[point1] + prices[point2]) / 2
            target_price = neckline + height
            
        confidence = self._calculate_pattern_confidence(prices, points)
        
        return DetectedPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            start_date=str(dates[point1].date()),
            end_date=str(dates[point2].date()),
            points=points,
            neckline=neckline,
            target_price=target_price,
            description=f"{pattern_type.value.replace('_', ' ').title()} pattern"
        )
        
    def _detect_triangles(self, data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, price_column: str):
        """Detect Triangle patterns (Ascending, Descending, Symmetrical)."""
        if len(peaks) < 2 or len(valleys) < 2:
            return
            
        prices = data[price_column].values
        
        # Look for sequences of peaks and valleys that form triangles
        min_points = 4  # At least 2 peaks and 2 valleys
        
        for i in range(len(data) - self.min_pattern_length):
            window_end = min(i + self.min_pattern_length * 2, len(data))
            window_peaks = peaks[(peaks >= i) & (peaks < window_end)]
            window_valleys = valleys[(valleys >= i) & (valleys < window_end)]
            
            if len(window_peaks) >= 2 and len(window_valleys) >= 2:
                triangle_type = self._identify_triangle_type(prices, window_peaks, window_valleys)
                if triangle_type:
                    pattern = self._create_triangle_pattern(data, window_peaks, window_valleys, triangle_type)
                    if pattern:
                        self.detected_patterns.append(pattern)
                        
    def _identify_triangle_type(self, prices: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> Optional[PatternType]:
        """Identify the type of triangle pattern."""
        if len(peaks) < 2 or len(valleys) < 2:
            return None
            
        # Calculate trend lines
        peak_slope = self._calculate_slope(peaks, prices[peaks])
        valley_slope = self._calculate_slope(valleys, prices[valleys])
        
        # Define thresholds for slope significance
        slope_threshold = 0.001
        
        if abs(peak_slope) < slope_threshold and valley_slope > slope_threshold:
            return PatternType.ASCENDING_TRIANGLE
        elif peak_slope < -slope_threshold and abs(valley_slope) < slope_threshold:
            return PatternType.DESCENDING_TRIANGLE
        elif peak_slope < -slope_threshold and valley_slope > slope_threshold:
            return PatternType.SYMMETRICAL_TRIANGLE
            
        return None
        
    def _calculate_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate the slope of a line through the given points."""
        if len(x) < 2:
            return 0
        return (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
        
    def _create_triangle_pattern(self, data: pd.DataFrame, peaks: np.ndarray, 
                               valleys: np.ndarray, triangle_type: PatternType) -> Optional[DetectedPattern]:
        """Create a triangle pattern object."""
        prices = data[data.columns[3]].values
        dates = data.index
        
        # Combine and sort all points
        all_points = [(idx, prices[idx], 'peak') for idx in peaks] + \
                    [(idx, prices[idx], 'valley') for idx in valleys]
        all_points.sort(key=lambda x: x[0])
        
        if len(all_points) < 4:
            return None
            
        points = []
        for idx, price, point_type in all_points:
            points.append(PatternPoint(idx, price, str(dates[idx].date()), point_type))
            
        # Calculate pattern boundaries
        start_idx = all_points[0][0]
        end_idx = all_points[-1][0]
        
        confidence = self._calculate_pattern_confidence(prices, points)
        
        return DetectedPattern(
            pattern_type=triangle_type,
            confidence=confidence,
            start_date=str(dates[start_idx].date()),
            end_date=str(dates[end_idx].date()),
            points=points,
            description=f"{triangle_type.value.replace('_', ' ').title()} pattern"
        )
        
    def _detect_cup_and_handle(self, data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, price_column: str):
        """Detect Cup and Handle patterns."""
        if len(valleys) < 3 or len(peaks) < 2:
            return
            
        prices = data[price_column].values
        dates = data.index
        
        # Look for U-shaped valleys (cup) followed by smaller pullback (handle)
        for i in range(len(valleys) - 2):
            # Potential cup formation with 3 valleys
            left_rim = valleys[i]
            bottom = valleys[i + 1]
            right_rim = valleys[i + 2]
            
            if self._is_cup_formation(prices, left_rim, bottom, right_rim):
                # Look for handle formation after the cup
                handle_start = right_rim
                handle_peaks = peaks[peaks > handle_start]
                handle_valleys = valleys[valleys > handle_start]
                
                if len(handle_peaks) > 0 and len(handle_valleys) > 0:
                    handle_peak = handle_peaks[0]
                    handle_valley = handle_valleys[0] if len(handle_valleys) > 0 else None
                    
                    if handle_valley and self._is_handle_formation(prices, right_rim, handle_peak, handle_valley):
                        pattern = self._create_cup_and_handle_pattern(
                            data, left_rim, bottom, right_rim, handle_peak, handle_valley
                        )
                        self.detected_patterns.append(pattern)
                        
    def _is_cup_formation(self, prices: np.ndarray, left: int, bottom: int, right: int) -> bool:
        """Check if points form a valid cup formation."""
        # Bottom should be significantly lower than rims
        rim_avg = (prices[left] + prices[right]) / 2
        depth = rim_avg - prices[bottom]
        
        # Cup should be at least 10% deep
        if depth / rim_avg < 0.10:
            return False
            
        # Rims should be roughly at the same level
        rim_tolerance = 0.05
        if abs(prices[left] - prices[right]) / prices[left] > rim_tolerance:
            return False
            
        return True
        
    def _is_handle_formation(self, prices: np.ndarray, cup_rim: int, handle_peak: int, handle_valley: int) -> bool:
        """Check if points form a valid handle formation."""
        # Handle should be a relatively small pullback
        handle_depth = prices[handle_peak] - prices[handle_valley]
        cup_rim_price = prices[cup_rim]
        
        # Handle depth should be less than 1/3 of cup rim height
        return handle_depth / cup_rim_price < 0.15
        
    def _create_cup_and_handle_pattern(self, data: pd.DataFrame, left_rim: int, bottom: int, 
                                     right_rim: int, handle_peak: int, handle_valley: int) -> DetectedPattern:
        """Create a cup and handle pattern object."""
        prices = data[data.columns[3]].values
        dates = data.index
        
        points = [
            PatternPoint(left_rim, prices[left_rim], str(dates[left_rim].date()), "left_rim"),
            PatternPoint(bottom, prices[bottom], str(dates[bottom].date()), "cup_bottom"),
            PatternPoint(right_rim, prices[right_rim], str(dates[right_rim].date()), "right_rim"),
            PatternPoint(handle_peak, prices[handle_peak], str(dates[handle_peak].date()), "handle_peak"),
            PatternPoint(handle_valley, prices[handle_valley], str(dates[handle_valley].date()), "handle_valley")
        ]
        
        # Calculate target price (cup depth added to breakout point)
        cup_depth = (prices[left_rim] + prices[right_rim]) / 2 - prices[bottom]
        target_price = prices[handle_peak] + cup_depth
        
        confidence = self._calculate_pattern_confidence(prices, points)
        
        return DetectedPattern(
            pattern_type=PatternType.CUP_AND_HANDLE,
            confidence=confidence,
            start_date=str(dates[left_rim].date()),
            end_date=str(dates[handle_valley].date()),
            points=points,
            target_price=target_price,
            description="Cup and Handle pattern"
        )
        
    def _detect_flags_and_pennants(self, data: pd.DataFrame, peaks: np.ndarray, valleys: np.ndarray, price_column: str):
        """Detect Flag and Pennant patterns."""
        # Flags and pennants are short-term continuation patterns
        # Look for consolidation after strong moves
        
        prices = data[price_column].values
        
        # Look for strong price moves followed by consolidation
        for i in range(20, len(data) - 20):  # Need space before and after
            # Check for strong move (flagpole)
            move_start = max(0, i - 20)
            move_end = i
            
            price_change = abs(prices[move_end] - prices[move_start])
            avg_price = (prices[move_end] + prices[move_start]) / 2
            
            # Strong move threshold (at least 5% change)
            if price_change / avg_price > 0.05:
                # Look for consolidation pattern after the move
                consolidation_end = min(len(data), i + 15)
                consolidation_peaks = peaks[(peaks > move_end) & (peaks < consolidation_end)]
                consolidation_valleys = valleys[(valleys > move_end) & (valleys < consolidation_end)]
                
                if len(consolidation_peaks) >= 2 and len(consolidation_valleys) >= 2:
                    pattern_type = self._identify_flag_or_pennant(prices, consolidation_peaks, consolidation_valleys)
                    if pattern_type:
                        pattern = self._create_flag_pattern(data, move_start, move_end, 
                                                          consolidation_peaks, consolidation_valleys, pattern_type)
                        if pattern:
                            self.detected_patterns.append(pattern)
                            
    def _identify_flag_or_pennant(self, prices: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> Optional[PatternType]:
        """Identify if consolidation is a flag or pennant."""
        if len(peaks) < 2 or len(valleys) < 2:
            return None
            
        # Check if peaks and valleys are roughly parallel (flag) or converging (pennant)
        peak_slope = self._calculate_slope(peaks, prices[peaks])
        valley_slope = self._calculate_slope(valleys, prices[valleys])
        
        # If slopes are similar, it's a flag; if converging, it's a pennant
        slope_diff = abs(peak_slope - valley_slope)
        
        if slope_diff < 0.001:
            return PatternType.FLAG
        else:
            return PatternType.PENNANT
            
    def _create_flag_pattern(self, data: pd.DataFrame, move_start: int, move_end: int,
                           peaks: np.ndarray, valleys: np.ndarray, pattern_type: PatternType) -> Optional[DetectedPattern]:
        """Create a flag or pennant pattern object."""
        prices = data[data.columns[3]].values
        dates = data.index
        
        points = [PatternPoint(move_start, prices[move_start], str(dates[move_start].date()), "flagpole_start")]
        points.append(PatternPoint(move_end, prices[move_end], str(dates[move_end].date()), "flagpole_end"))
        
        # Add consolidation points
        for peak in peaks:
            points.append(PatternPoint(peak, prices[peak], str(dates[peak].date()), "consolidation_peak"))
        for valley in valleys:
            points.append(PatternPoint(valley, prices[valley], str(dates[valley].date()), "consolidation_valley"))
            
        # Sort points by date
        points.sort(key=lambda x: x.index)
        
        if len(points) < 4:
            return None
            
        # Calculate target price (flagpole height projected from breakout)
        flagpole_height = abs(prices[move_end] - prices[move_start])
        breakout_price = prices[move_end]
        
        if prices[move_end] > prices[move_start]:  # Uptrend
            target_price = breakout_price + flagpole_height
        else:  # Downtrend
            target_price = breakout_price - flagpole_height
            
        confidence = self._calculate_pattern_confidence(prices, points)
        
        return DetectedPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            start_date=str(dates[move_start].date()),
            end_date=str(dates[points[-1].index].date()),
            points=points,
            target_price=target_price,
            description=f"{pattern_type.value.title()} pattern"
        )
        
    def _calculate_pattern_confidence(self, prices: np.ndarray, points: List[PatternPoint]) -> float:
        """Calculate confidence score for a detected pattern."""
        if len(points) < 3:
            return 0.5
            
        # Base confidence on various factors
        confidence = 0.7  # Base confidence
        
        # Factor 1: Pattern symmetry (for applicable patterns)
        # Factor 2: Volume confirmation (if available)
        # Factor 3: Pattern completion
        # Factor 4: Time span (longer patterns are more reliable)
        
        time_span = points[-1].index - points[0].index
        if time_span > 30:  # Patterns spanning more than 30 periods
            confidence += 0.1
        elif time_span < 10:  # Very short patterns are less reliable
            confidence -= 0.1
            
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))
        
    def get_pattern_summary(self) -> Dict:
        """Get summary of all detected patterns."""
        if not self.detected_patterns:
            return {"total_patterns": 0, "patterns_by_type": {}}
            
        patterns_by_type = {}
        for pattern in self.detected_patterns:
            pattern_name = pattern.pattern_type.value
            if pattern_name not in patterns_by_type:
                patterns_by_type[pattern_name] = 0
            patterns_by_type[pattern_name] += 1
            
        return {
            "total_patterns": len(self.detected_patterns),
            "patterns_by_type": patterns_by_type,
            "average_confidence": sum(p.confidence for p in self.detected_patterns) / len(self.detected_patterns)
        }


# Example usage
if __name__ == "__main__":
    # This would be used with actual data from data_loader
    print("Pattern Detection Module - Ready for integration")
    
    # Example of creating a detector
    detector = PatternDetector(min_pattern_length=20, peak_prominence=0.02)
    print(f"Detector initialized with min_pattern_length={detector.min_pattern_length}")
    print(f"Supported patterns: {[pt.value for pt in PatternType]}")