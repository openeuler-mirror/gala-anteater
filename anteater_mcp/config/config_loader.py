# Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
"""配置文件处理模块"""
import toml
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from pathlib import Path
from copy import deepcopy
import sys
import os
# 从当前文件位置向上两级到达项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
can_import = True
try:
    from apps.schemas.config import ConfigModel as FrameworkConfigModel
except ImportError as e:
    can_import = False

# 然后使用绝对导入


class LanguageEnum(str, Enum):
    """语言枚举"""
    ZH = "zh"
    EN = "en"


class RemoteConfigModel(BaseModel):
    """远程配置模型"""
    name: str = Field(..., description="远程主机名称")
    os_type: str = Field(..., description="远程主机操作系统类型")
    host: str = Field(..., description="远程主机地址")
    port: int = Field(..., description="远程主机端口")
    username: str = Field(..., description="远程主机用户名")
    password: str = Field(..., description="远程主机密码")


class PublicConfigModel(BaseModel):
    """公共配置模型"""
    language: LanguageEnum = Field(default=LanguageEnum.ZH, description="语言")
    remote_hosts: list[RemoteConfigModel] = Field(default=[], description="远程主机列表")
    llm_remote: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", description="LLM远程主机地址")
    llm_model: str = Field(default="qwen3-coder-480b-a35b-instruct", description="LLM模型名称")
    llm_api_key: str = Field(default="", description="LLM API Key")
    max_tokens: int = Field(default=8192, description="LLM最大Token数")
    temperature: float = Field(default=0.7, description="LLM温度参数")


class ConfigModel(BaseModel):
    """公共配置模型"""
    public_config: PublicConfigModel = Field(default=PublicConfigModel(), description="公共配置")
    private_config: Any = Field(default=None, description="私有配置")


class GalaAnteaterConfig():
    """配置文件读取和使用Class"""

    def __init__(self) -> None:
        """读取配置文件；当PROD环境变量设置时，配置文件将在读取后删除"""
        config_file = os.path.join("config", "public", "public_config.toml")
        self._config = ConfigModel()
        self._config.public_config = PublicConfigModel.model_validate(toml.load(config_file))
        framework_config_file = os.getenv("CONFIG")
        if framework_config_file is None:
            if can_import:
                framework_config_file = os.path.join("..", "config", "config.toml")
        if framework_config_file and os.path.exists(framework_config_file):
            framework_config = FrameworkConfigModel.model_validate(toml.load(framework_config_file))
            self._config.public_config.llm_remote = framework_config.llm.endpoint
            self._config.public_config.llm_model = framework_config.llm.model
            self._config.public_config.llm_api_key = framework_config.llm.key
            self._config.public_config.max_tokens = framework_config.llm.max_tokens
            self._config.public_config.temperature = framework_config.llm.temperature
    def load_private_config(self) -> None:
        """加载私有配置文件"""
        pass

    def get_config(self) -> ConfigModel:
        """获取配置文件内容"""
        return deepcopy(self._config)
