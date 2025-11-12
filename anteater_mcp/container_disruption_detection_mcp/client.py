import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from enum import Enum
from typing import Union

from mcp import ClientSession
from mcp.client.sse import sse_client

from .utils import load_kpis_from_job

logger = logging.getLogger(__name__)


class MCPStatus(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MCPClient:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self.client: Union[ClientSession, None] = None
        self.status = MCPStatus.UNINITIALIZED

    async def _main_loop(self) -> None:
        try:
            client = sse_client(url=self.url, headers=self.headers)
        except Exception as e:
            self.error_sign.set()
            raise Exception(f"创建Client失败: {e}")

        try:
            exit_stack = AsyncExitStack()
            read, write = await exit_stack.enter_async_context(client)
            self.client = ClientSession(read, write)
            session = await exit_stack.enter_async_context(self.client)
            await session.initialize()
        except Exception as e:
            self.error_sign.set()
            self.status = MCPStatus.STOPPED
            raise Exception(f"初始化Client失败，错误信息：{e}")

        self.ready_sign.set()
        self.status = MCPStatus.RUNNING
        await self.stop_sign.wait()

        try:
            await exit_stack.aclose()
            self.status = MCPStatus.STOPPED
        except Exception as e:
            print(f"关闭Client失败，错误信息：{e}")

    async def init(self) -> None:
        self.ready_sign = asyncio.Event()
        self.error_sign = asyncio.Event()
        self.stop_sign = asyncio.Event()
        self.task = asyncio.create_task(self._main_loop())

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(self.ready_sign.wait()),
                asyncio.create_task(self.error_sign.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self.error_sign.is_set():
            self.status = MCPStatus.ERROR
            raise Exception("MCP Client 初始化失败")

    async def call_tool(self, tool_name: str, params: dict):
        return await self.client.call_tool(tool_name, params)

    async def stop(self) -> None:
        self.stop_sign.set()
        try:
            await self.task
        except Exception as e:
            print(f"关闭MCP Client失败，错误信息：{e}")


def _to_payload(x):
    """递归序列化对象为 JSON 可传输格式"""
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    if hasattr(x, "__dict__") and not isinstance(x, (str, bytes)):
        return {
            k: _to_payload(v) for k, v in x.__dict__.items() if not k.startswith("_")
        }
    if isinstance(x, (list, tuple)):
        return [_to_payload(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_payload(v) for k, v in x.items()}
    return x


async def main() -> None:
    base_dir = os.path.dirname(__file__)
    job_path = os.path.join(base_dir, "../config/container_disruption.job.json")
    anteater_conf_path = os.path.join(base_dir, "../config/gala-anteater.yaml")

    kpis, window, extra = load_kpis_from_job(job_path)
    logger.info("配置加载成功，开始检测。")

    client = MCPClient(url="http://127.0.0.1:12345/sse", headers={})
    await client.init()

    params = {
        "kpis": _to_payload(kpis),
        "window": _to_payload(window),
        "extra": _to_payload(extra),
        "anteater_conf": anteater_conf_path,
        "metric_info": {},
    }

    print(">>> 调用 container_disruption_detection_tool ...")
    result = await client.call_tool("container_disruption_detection_tool", params)

    anomalies_json = None
    if getattr(result, "content", None):
        for c in result.content:
            if getattr(c, "text", None):
                anomalies_json = c.text
                break

    anomalies_json = anomalies_json or "{}"
    print("=== 检测结果(JSON) ===")
    print(anomalies_json)

    try:
        parsed_result = json.loads(anomalies_json)
        is_anomaly = parsed_result.get("is_anomaly", False)
    except Exception as e:
        logger.error(f"检测结果解析失败: {e}")
        parsed_result = {}
        is_anomaly = False

    if is_anomaly:
        print("\n>>> 调用 root_cause_analysis_tool ...")
        analysis = await client.call_tool(
            "root_cause_analysis_tool", {"anomalies": parsed_result}
        )
        print("=== 根因分析结果 ===")
        print(analysis)
        source_data = None
        if getattr(analysis, "content", None):
            for c in analysis.content:
                if getattr(c, "text", None):
                    source_data = c.text
                    break
        report_type = "anomaly"
    else:
        print("\n检测结果正常，无需根因分析。")
        source_data = anomalies_json
        report_type = "normal"

    print("\n>>> 调用 report_tool ...")
    if not source_data:
        logger.error("未找到有效的 source_data，无法生成报告。")
        return

    try:
        payload = {
            "source_data": json.loads(source_data),
            "report_type": report_type,
        }
    except Exception:
        payload = {
            "source_data": parsed_result,
            "report_type": report_type,
        }

    report = await client.call_tool("report_tool", payload)
    print("=== 报告 ===")
    print(report)

    await asyncio.sleep(0.5)
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
