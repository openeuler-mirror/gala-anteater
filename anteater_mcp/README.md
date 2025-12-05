# Container Disruption Detection MCP

## 启动方式

```bash
git clone https://gitee.com/openeuler/gala-anteater
cd gala-anteater
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m anteater_mcp.container_disruption_detection_mcp.mcp_server

默认监听地址：`0.0.0.0:12345`