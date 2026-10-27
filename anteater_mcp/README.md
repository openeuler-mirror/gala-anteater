# Container Disruption Detection MCP

## 前置条件

MCP Server 运行依赖以下配置文件，需先安装到 `/etc/gala-anteater-mcp/config/` 目录：

- `gala-anteater.yaml`：数据源（Prometheus/ArangoDB 等）及大模型（`llm_config`）连接参数；
- `container_disruption.job.json`：检测任务及 KPI 配置；
- `suggestion_generation_prompts.json`：恢复建议生成使用的 prompt 模板。

上述文件位于本工程 `config/` 目录，可通过下方"安装为服务"方式自动拷贝，也可手工复制到该路径后再启动。

## 启动方式

### 方法一：源码手工启动（适用于开发者）

```bash
git clone https://atomgit.com/openeuler/gala-anteater
cd gala-anteater
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m anteater_mcp.container_disruption_detection_mcp.mcp_server
```

### 方法二：安装为服务（适用于生产环境）

在 `anteater_mcp/` 目录下执行安装命令，会同时拷贝配置文件并注册 systemd 服务：

```bash
python3 setup.py install
systemctl start container-disruption-detection-mcpserver
```

安装后可直接通过命令入口启动（与 systemd 服务等价）：

```bash
container-disruption-detection-mcp
```

## 监听地址

默认监听地址：`0.0.0.0:12345`
