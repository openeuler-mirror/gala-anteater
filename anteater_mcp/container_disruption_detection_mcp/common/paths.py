# -*- coding: utf-8 -*-
"""
MCP Server 配置文件路径的唯一定义处

服务运行时读取的每个配置文件都由这里的 MCP_CONFIG_DIR 派生，安装产物
（anteater_mcp/setup.py 的 data_files）、读取代码与单测三处对齐同一份命名，
避免此前"代码读 prompts.json、安装的是 suggestion_generation_prompts.json"
这类文件名/目录漂移再次发生。
"""

import os

# setup.py 将配置文件安装到的目录；未显式指定时固定为系统配置目录
DEFAULT_MCP_CONFIG_DIR = os.path.join("/etc", "gala-anteater-mcp", "config")

# 源码直接运行、容器部署与单测可用该环境变量指向其它目录，
# 无需把配置写进宿主机的 /etc
MCP_CONFIG_DIR = (
    os.environ.get("ANTEATER_MCP_CONFIG_DIR") or DEFAULT_MCP_CONFIG_DIR
)

# 配置目录下的文件名，须与 anteater_mcp/config/ 及 setup.py data_files 一致
MAIN_CONFIG_FILE = "gala-anteater.yaml"
JOB_CONFIG_FILE = "container_disruption.job.json"
PROMPT_CONFIG_FILE = "suggestion_generation_prompts.json"


def _config_path(file_name):
    """配置目录下某个文件的绝对路径"""
    return os.path.join(MCP_CONFIG_DIR, file_name)


def main_config_path():
    """主配置 gala-anteater.yaml 的路径"""
    return _config_path(MAIN_CONFIG_FILE)


def job_config_path():
    """检测任务配置 container_disruption.job.json 的路径"""
    return _config_path(JOB_CONFIG_FILE)


def prompt_config_path():
    """恢复建议 prompt 模板 suggestion_generation_prompts.json 的路径"""
    return _config_path(PROMPT_CONFIG_FILE)
