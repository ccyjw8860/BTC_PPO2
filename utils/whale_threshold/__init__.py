"""
Whale Trade Threshold Calculator Module

This module provides functionality to calculate dynamic 99.5th percentile thresholds
for whale trades using a 24-hour sliding window on Bitcoin futures data.
"""

from .models import ThresholdDocument, WindowStats, ThresholdCalculationError
from .sliding_window import SlidingWindowBuffer
from .threshold_calculator import ThresholdCalculator
from .mongo_aggregator import MongoAggregator
from .batch_processor import BatchProcessor

__all__ = [
    'SlidingWindowBuffer',
    'ThresholdCalculator',
    'MongoAggregator',
    'BatchProcessor',
    'ThresholdDocument',
    'WindowStats',
    'ThresholdCalculationError',
]
