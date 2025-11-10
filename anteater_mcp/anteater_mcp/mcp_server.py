from typing import Union
import json
from mcp.server import FastMCP
from anteater.utils.log import logger
from typing import Dict, List, Tuple
from anteater_mcp.anteater_mcp.disruption_detector_api import find_discruption_source
from anteater.core.anomaly import Anomaly, RootCause
from config.config_loader import GalaAnteaterConfig

# 仅在 Linux 环境下强制使用 spawn 方式
import multiprocessing
import os
import sys
from anteater.core.ts import TimeSeries

if os.name == "posix":  # posix 表示 Linux/macOS
    multiprocessing.set_start_method("spawn", force=True)
# 创建MCP Server
mcp = FastMCP("Gala-anteater MCP Server", host="0.0.0.0", port=12400)

@mcp.prompt(description="调用逻辑:1. 当用户询问特定任务ID的机器性能是否劣化时调用。2. 检测结果将决定后续流程走向。\
            3. 调用完成后如果出现劣化现象，则把当前工具得到的结果作为入参，调用slow_node_detection_tool方法 ，如果没有出现劣化现象，则调用报告工具返回报告给用户。\
            4. 本方法得到的结果必须再调用generate_report 生成报告给到用户"
            )
@mcp.tool(
    name="disturbance_correlation_analysis_tool"
)
def disturbance_correlation_analysis_tool(victim_ts: TimeSeries, all_ts: List[TimeSeries]) -> List[RootCause]:
    """
    根据采集的指标时间序列，分析容器干扰关联情况
    入参 victim_ts ，如 192.168.2.122;
    返回 List[RootCause] 如果is_anomaly=false，该结果需要调用generate_report_tool再返回给用户;如果is_anomaly=True,该结果必须调用slow_node_detection_tool得到报告
    """
    logger.info("调用容器干扰关联分析工具")
    logger.info("victim_ts = " + str(victim_ts))

    root_causes = []
    root_causes = find_discruption_source(victim_ts, all_ts)
    return root_causes

def main():
    # 初始化并启动服务
    mcp.run(transport='sse')


if __name__ == "__main__":
    main()
