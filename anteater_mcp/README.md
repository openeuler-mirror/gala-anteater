# Container Disruption Detection MCP

## 目录结构

```
anteater_mcp/
├── config
│   ├── container_disruption.job.json
│   └── gala-anteater.yaml
├── container_disruption_detection_mcp
│   ├── api
│   │   ├── disruption_source_api.py
│   ├── client.py
│   ├── disruption_detector_api.py
│   ├── __init__.py
│   ├── mcp_data.py
│   ├── mcp_server.py
│   └── utils.py
├── __init__.py
├── log.settings.ini
├── README.md
├── service
│   └── container- disruption- detection-mcpserver.service
└── setup.py
```

## 启动方式

```bash
conda activate anteater
cd /home/mengzi/gala-anteater
export PYTHONPATH=$PYTHONPATH:/home/mengzi/gala-anteater

# 启动 MCP 服务
python -m anteater_mcp.container_disruption_detection_mcp.mcp_server

# 启动客户端（功能验证）
python -m anteater_mcp.container_disruption_detection_mcp.client
```

默认监听地址：`0.0.0.0:12345`