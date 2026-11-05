# Container Disruption Detection MCP

## Prerequisites

The MCP Server depends on the following configuration files, which must be installed to the `/etc/gala-anteater-mcp/config/` directory:

- `gala-anteater.yaml`: data source (Prometheus/ArangoDB, etc.) and LLM (`llm_config`) connection parameters;
- `container_disruption.job.json`: detection task and KPI configuration;
- `suggestion_generation_prompts.json`: prompt templates used for recovery suggestion generation.

These files are located in the `config/` directory of this project. They can be copied automatically by the "Install as a service" method below, or copied manually to that path before starting.

The configuration directory defaults to `/etc/gala-anteater-mcp/config/`, and can be pointed somewhere else (for example a configuration directory mounted inside a container) with the `ANTEATER_MCP_CONFIG_DIR` environment variable. The directory and the file names are defined once in `container_disruption_detection_mcp/common/paths.py` and are kept in sync with what `setup.py` installs (checked by `tests/unit_tests/test_mcp_config_paths.py`).

## Startup Method

### Method 1: Run from source (for developers)

```bash
git clone https://atomgit.com/openeuler/gala-anteater
cd gala-anteater
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m anteater_mcp.container_disruption_detection_mcp.mcp_server
```

### Method 2: Install as a service (for production)

Run the install command in the `anteater_mcp/` directory, which copies the configuration files and registers the systemd service:

```bash
python3 setup.py install
systemctl start container-disruption-detection-mcpserver
```

After installation, you can also start it directly through the command entry point (equivalent to the systemd service):

```bash
container-disruption-detection-mcp
```

## Listening Address

Default listening address: `0.0.0.0:12345`
