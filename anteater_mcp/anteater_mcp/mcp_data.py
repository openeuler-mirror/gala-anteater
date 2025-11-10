from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
class TimeSeries:
    """The time series class"""
            self,
            metric: str,
            labels: Dict,
            time_stamps: List[int],
            values: List[float]):
        """The time series initializer"""
    metric: str = Field(default = "", description="检测指标名称")
    labels: Dict = File = labels.copy()
    time_stamps = time_stamps.copy()
    values = values.copy()