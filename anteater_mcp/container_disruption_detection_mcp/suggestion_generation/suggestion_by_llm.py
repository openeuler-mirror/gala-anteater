# server.py

from fastmcp import FastMCP
import asyncio

from pydantic import BaseModel
from typing import Optional
import json
import os
import yaml
from json_repair import repair_json
from openai import OpenAI




class ToolMeta(BaseModel):
    name: str
    description: str


TOOL_DETECTION = ToolMeta(name="container_disruption_detection_tool", description="xxx")
TOOL_RCA = ToolMeta(name="container_interference_analysis_tool", description="xxx")
TOOL_RECOVERY = ToolMeta(name="container_interference_recovery_suggestion_tool",
                         description="This tool generate suggestions about how to recovery from"
                                     " key container interference.\n"
                                     "[Input] This tool should be provides with\n"
                                     f" 1. the <detection_report> of *{TOOL_DETECTION.name}*\n"
                                     f" 2. the <analysis_report> of *{TOOL_RCA.name}*\n"
                                     f"[Output] This tool generates the a list of jsons, "
                                     f"each consisting of three element about the suggestion:"
                                     f"the **suggestion** itself, "
                                     f"the **evidence** of providing this suggestion,"
                                     f"and an **example** command of following the suggestion.")

class LLMConfig(BaseModel):
    base_url: str = None
    api_key: str = None
    model: str = None
    timeout: Optional[int] = 10


class LLMTool:
    def __init__(self, llm_config: LLMConfig):
        default_config_path = self.get_default_config_path()
        default_config: dict = yaml.load(open("run/test.yaml"), Loader=yaml.SafeLoader)
        default_config: LLMConfig = LLMConfig(**default_config["llm_config"])
        self._llm_config = default_config.model_copy(
            update=llm_config.model_dump(exclude_unset=True, exclude_none=True), deep=True)

    @classmethod
    def get_default_config_path(cls):
        return os.path.join("anteater_mcp", "config", "gala-anteater.yaml")

    def get_client(self, base_url=None, api_key=None):
        if base_url is None:
            base_url = self._llm_config.base_url
        if api_key is None:
            api_key = self._llm_config.api_key
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        return client

    async def _query(self, system_prompt: str, user_prompt: str):
        client = self.get_client()
        stream = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            model=self._llm_config.model,
            stream=True,
            timeout=10
        )
        ans = []
        for chunk in stream:
            chunk_ans = chunk.choices[0].delta.content
            if chunk_ans:
                ans.append(chunk_ans)
        return ''.join(ans)

    def get_available_models(self):
        client = self.get_client()
        return client.models.list()

    async def query_json(self, user_prompt, format_example, system_prompt="", max_retry=3):
        system_prompt = (f"{system_prompt} [Answer Format] **please answer with a json**, example(s) is/are as follows:"
                         f" {format_example}")
        for i in range(max_retry):
            try:
                ans = await self._query(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                ans = repair_json(ans)
                if ans:
                    ans = json.loads(ans)
                    return ans
            except Exception as e:
                print(f"Query LLM failed [{i + 1}/{max_retry}]: {e}")
                continue
        raise Exception(f"Query LLM failed: [User Prompt]{user_prompt} [System Prompt]{system_prompt}]")


class DetectionReport(BaseModel):
    class Detail(BaseModel):
        container_id: str
        metric_id: str
        start_timestamp: int
        end_timestamp: int
        abnormal_level: str
        disruption_peak: int

    metric_list: list[str]
    disruption_cnt: int
    details: list[Detail]


class RCAReport(BaseModel):
    class Detail(BaseModel):
        container_id: str
        disrupted_metric_id: str
        interf_src_probs: dict[str, float] = {}

    metric_list: list[str]
    disruption_cnt: int
    details: list[Detail]


class RecoverySuggestion(BaseModel):
    suggestion: str
    evidence: str
    example: str


class Output(BaseModel):
    task_id: str = None
    code: int
    msg: str
    detection_report: Optional[DetectionReport] = None
    analysis_report: Optional[RCAReport] = None
    recovery_suggestion: Optional[list[RecoverySuggestion]] = None


async def naive_recovery_suggestion_llm(task_id: str, detection_report: dict | DetectionReport,
                                        analysis_report: dict | RCAReport) -> Output:
    """
    (极简版本)获取干扰恢复建议
    :param task_id: 任务id，供未来提效和提高准确性（顺序访问下，detection_report和analysis_report可以读取而不需要传回）
    :param detection_report: 检测工具的输出
    :param analysis_report: 干扰分析工具的输出
    :return: 输出一个Output
    """
    # 格式处理
    try:
        if isinstance(detection_report, dict):
            detection_report: DetectionReport = DetectionReport.model_validate(detection_report)
        if isinstance(analysis_report, dict):
            analysis_report: RCAReport = RCAReport.model_validate(analysis_report)
    except:
        print("输入数据结构不正确，请检查detection_report和analysis_report是否符合约定")  # TODO: 可以有其他记录方式
        return Output(task_id=task_id, code=400,
                      msg=f"[{TOOL_RECOVERY.name}] 输入数据结构不正确："
                          f"请检查detection_report和analysis_report是否符合约定")
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    try:
        prompts = json.load(open(prompt_path, encoding="utf8"))
        user_prompt = prompts["user_prompt"].format(detection_report=detection_report.model_dump_json(),
                                                    analysis_report=analysis_report.model_dump_json())
        format_examples = prompts["format_examples"]
    except Exception as e:
        return Output(task_id=task_id, code=400,
                      msg=f"Prompt模板读取失败 {e}\n"
                          f"模板文件 {prompt_path} 可能损坏",
                      recovery_suggestion=[])
    try:
        config = LLMConfig()
        tool = LLMTool(config)
        tool.get_available_models()
    except Exception as e:
        return Output(task_id=task_id, code=400,
                      msg=f"访问LLM初始化失败 {e}\n"
                          f"请检查配置文件 {LLMTool.get_default_config_path()}",
                      recovery_suggestion=[])
    try:
        result = await tool.query_json(user_prompt, json.dumps(format_examples))
    except Exception as e:
        return Output(task_id=task_id, code=400, msg=f"[{TOOL_RECOVERY.name}] 调用大模型出错：{e}")

    results: list[RecoverySuggestion] = []
    for item in result:
        try:
            item_suggestion = RecoverySuggestion.model_validate(item)
            results.append(item_suggestion)
        except Exception as e:
            continue
    return Output(task_id=task_id, code=200, msg="success", recovery_suggestion=results)

__all__ = [
    naive_recovery_suggestion_llm
]
