# Container Disruption Detection MCP

基于 **Gala Gopher** 时序引擎与 **Gala Anteater** 算法框架，实现容器级异常检测、根因分析（RCA）与性能干扰诊断报告生成。

## 功能概述

* 基于 SPOT / TS-DBSCAN 的容器异常检测
* 基于相关性分析的容器根因定位
* 输出结构化 JSON 报告（由大模型生成最终自然语言报告）

## 目录结构

```
anteater_mcp/
├── config/
│   ├── container_disruption.job.json
│   └── gala-anteater.yaml
│
├── container_disruption_detection_mcp/
│   ├── mcp_server.py                # MCP 服务主程序
│   ├── mcp_data.py                  # 数据结构定义
│   ├── report_api.py                # 报告生成逻辑（结构化输出）
│   ├── utils.py                     # 通用工具
│   ├── client.py                    # 客户端示例
│   └── disruption_detector_api.py   # 检测接口封装
│
├── service/
│   └── container-disruption-detection-mcpserver.service
│
├── log.settings.ini
├── setup.py
└── README.md
```

## 启动方式

```bash
cd /home/mengzi/gala-anteater
export PYTHONPATH=$PYTHONPATH:/home/mengzi/gala-anteater

# 启动 MCP 服务
python -m anteater_mcp.container_disruption_detection_mcp.mcp_server

# 启动客户端（功能验证）
python -m anteater_mcp.container_disruption_detection_mcp.client
```

默认监听地址：`0.0.0.0:12345`

## MCP 工具接口

| 工具名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `container_disruption_detection_tool` | 容器异常检测 | KPI 参数、时间窗口、配置等 | `PerceptionResult` |
| `root_cause_analysis_tool` | 根因分析 | 异常检测结果 | `RootCauseResult` |
| `report_tool` | 报告生成 | 检测或根因结果 + 报告类型 | JSON 报告 |

### 1. 容器异常检测

```python
container_disruption_detection_tool(
    kpis: List[KPIParam],
    window: WindowParam = WindowParam(),
    extra: Optional[ExtraConfig] = None,
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: Optional[str] = None
) -> PerceptionResult
```

说明：

* 自动发现活跃机器 ID；
* 输出是否存在异常及异常详情。

### 2. 根因分析

```python
root_cause_analysis_tool(
    anomalies: PerceptionResult,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str = ""
) -> RootCauseResult
```

说明：

* 仅在 `is_anomaly=True` 时调用；
* 基于多维时序分析定位潜在干扰源。



### 3. 报告生成

```python
report_tool(
    source_data: Union[PerceptionResult, RootCauseResult],
    report_type: ReportType
) -> dict
```

说明：

* 根据检测或根因分析结果生成结构化 JSON 报告；
* 不生成自然语言文本，由 LLM 负责最终渲染；
* 内部调用 `generate_normal_report()` 或 `generate_degraded_report()`。