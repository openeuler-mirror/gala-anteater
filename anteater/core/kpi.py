
from dataclasses import dataclass, field


@dataclass
class KPI:
    metric: str = None
    kpi_type: str = None
    entity_name: str = None
    enable: bool = False
    description: str = ""
    parameter: dict = field(default=dict)

