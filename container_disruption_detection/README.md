# Container Disruption Detection MCP

基于 **Gala Gopher** 时序引擎与 **Gala Anteater** AI 算法，实现容器级异常检测、根因分析（RCA）与报告生成

## 功能概述

* 基于 SPOT 与 TS-DBSCAN 的容器异常检测
* 基于相关性的容器根因分析
* 自动生成 Markdown 格式诊断报告

## 目录结构

```
container_disruption_detection/
|— config/
|   |— container_disruption.job.json      # KPI 与模型配置
|   |— gala-anteater.yaml                 # Anteater 数据源配置
|
|— container_disruption_detection_mcp/
|   |— mcp_server.py                      # MCP 服务主程序
|   |— mcp_data.py                        # 数据结构定义
|   |— utils.py                           # 配置与任务加载工具
|   |— client.py                          # MCP 客户端示例
|
|— common/
|   |— loader.py                          # MetricLoader 构建逻辑
|
|— service/
|   |— container-disruption-detection-mcpserver.service
|
|— setup.py
|— README.md
```

## 启动方式

```bash
cd /home/mengzi/gala-anteater
export PYTHONPATH=$PYTHONPATH:/home/mengzi/gala-anteater
python -m container_disruption_detection.container_disruption_detection_mcp.mcp_server
python -m container_disruption_detection.container_disruption_detection_mcp.client
```

默认监听 `0.0.0.0:12345`

## 工具接口

| 工具名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `container_disruption_detection_tool` | 容器异常检测 | KPI 参数、时间窗口、配置等 | 异常列表（`List[AnomalyModel]`） |
| `root_cause_analysis_tool` | 根因分析 | 指标名、容器名、机器 ID | 根因列表（`List[RootCauseModel]`） |
| `generate_report_tool` | 报告生成 | 异常列表、报告类型 | Markdown 报告字典 |

### 1. 容器异常检测

**Tool:** `container_disruption_detection_tool`

```python
container_disruption_detection_tool(
    kpis: List[KPIParam],
    window: WindowParam = WindowParam(),
    extra: Optional[ExtraConfig] = None,
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: Optional[str] = None
) -> List[AnomalyModel]
```

**说明：**

* 自动发现活跃机器 ID（当未指定 `machine_id` 时）
* 若 `kpis` 为空，将抛出异常
* `extra.extra_metrics` 可用于传入额外监控指标
* 返回每个异常的详细信息，包括机器、指标、分数、标签及根因分析结果

### 2. 根因分析

**Tool:** `root_cause_analysis_tool`

```python
root_cause_analysis_tool(
    metric: str,
    victim_container_name: str,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str
) -> List[RootCauseModel]
```

**说明：**

* `machine_id` 为必填参数
* 若找不到指定容器的时序，将抛出异常
* 基于同机容器的相关性分析，返回最相关的前若干个根因指标（默认取前 3 个）

### 3. 报告生成

**Tool:** `generate_report_tool`

```python
generate_report_tool(anomalies: List[AnomalyModel], report_type: ReportType = ReportType.anomaly) -> Dict[str, str]
```

**说明：**

* 当 `report_type` 为 `ReportType.normal` 或异常列表为空时，返回“运行正常”报告
* 当存在异常时，生成详细的 Markdown 格式诊断报告
* 返回结构：`{"markdown": "<报告内容>"}`，可直接渲染为 Markdown