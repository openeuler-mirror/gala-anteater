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
cd servers/gala_anteater/container_disruption_detection/
python3 container_disruption_detection_mcp/mcp_server.py
```

默认监听 `0.0.0.0:12345`

## 工具接口

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
)
```

* 自动发现活跃机器 ID
* `outlier_ratio_th` 为异常比例阈值（默认 0.1）

### 2. 根因分析

**Tool:** `rca_tool`

```python
rca_tool(
    metric: str,
    victim_container_name: str,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str
)
```

计算同机容器间的相关性，返回前 3 个可能根因。

### 3. 报告生成

**Tool:** `report_tool`

```python
report_tool(anomalies: List[AnomalyModel], report_type: ReportType)
```

输出 Markdown 格式诊断报告。

## 依赖

* Python ≥ 3.8
* gala-anteater
* numpy, pandas, pydantic
* mcp