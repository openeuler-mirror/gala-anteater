import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from enum import Enum
from typing import Union

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger("container_disruption_client")
logging.basicConfig(level=logging.INFO)


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
            raise RuntimeError(f"创建 MCP SSE Client 失败: {e}")

        try:
            exit_stack = AsyncExitStack()
            read, write = await exit_stack.enter_async_context(client)

            self.client = ClientSession(read, write)
            session = await exit_stack.enter_async_context(self.client)

            await session.initialize()
        except Exception as e:
            self.error_sign.set()
            self.status = MCPStatus.STOPPED
            raise RuntimeError(f"初始化 MCP 客户端失败: {e}")

        self.ready_sign.set()
        self.status = MCPStatus.RUNNING

        # 等待关闭信号
        await self.stop_sign.wait()

        try:
            await exit_stack.aclose()
        except Exception as e:
            logger.warning(f"MCP Client 关闭异常: {e}")

        self.status = MCPStatus.STOPPED

    async def init(self) -> None:
        """启动 MCP 连接"""
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
            raise RuntimeError("MCP Client 初始化失败")

    async def call_tool(self, tool_name: str, params: dict):
        """调用任意 MCP 工具"""
        if not self.client:
            raise RuntimeError("MCP 客户端未初始化")

        return await self.client.call_tool(tool_name, params)

    async def stop(self) -> None:
        """关闭连接"""
        self.stop_sign.set()
        try:
            await self.task
        except Exception:
            pass


# 从 MCP 返回体提取 JSON
def extract_json_from_mcp(result) -> dict:
    """
    从 MCP 返回体中提取 JSON
    result.content: List[MessageContent]
    MessageContent 可能包含 text / markdown 多种格式
    """
    if not getattr(result, "content", None):
        return {}

    for c in result.content:
        text = getattr(c, "text", None) or getattr(c, "md", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except Exception:
            continue

    return {}


async def main() -> None:
    client = MCPClient(url="http://127.0.0.1:12345/sse", headers={})
    await client.init()

    task_id = "task-001"
    request_payload = {
        "task_id": task_id,
        "container_keyword_list": [],
        "metric_keyword_list": ["sli_container"],
        "analysis_timestamp": int(time.time() * 1000),
        "analysis_interval": 20 * 60 * 1000,  # 20 min
    }

    # 1. 容器干扰检测
    print("\n>>> 调用 container_disruption_detection_tool ...")

    detect_result = await client.call_tool(
        "container_disruption_detection_tool",
        {"request": json.dumps(request_payload)},
    )
    detect_json = extract_json_from_mcp(detect_result)

    print("\n=== 1) 检测结果 ===")
    print(json.dumps(detect_json, indent=2, ensure_ascii=False))

    if detect_json.get("code") != 200:
        print("检测失败，停止流程。")
        await client.stop()
        return

    detection_report = detect_json.get("detection_report", {})

    # 2. 容器干扰源分析
    print("\n>>> 调用 container_interference_analysis_tool ...")

    analysis_payload = {
        "task_id": task_id,
        "detection_report": detection_report,
    }

    analysis_result = await client.call_tool(
        "container_interference_analysis_tool",
        {"request": json.dumps(analysis_payload)},
    )
    analysis_json = extract_json_from_mcp(analysis_result)

    print("\n=== 2) 干扰分析结果 ===")
    print(json.dumps(analysis_json, indent=2, ensure_ascii=False))

    if analysis_json.get("code") != 200:
        print("干扰分析失败，停止流程。")
        await client.stop()
        return

    analysis_report = analysis_json.get("analysis_report", {})

    # 3. 干扰恢复建议
    print("\n>>> 调用 container_interference_recovery_suggestion_tool ...")

    recovery_payload = {
        "task_id": task_id,
        "detection_report": detection_report,
        "analysis_report": analysis_report,
    }

    recovery_result = await client.call_tool(
        "container_interference_recovery_suggestion_tool",
        {"request": json.dumps(recovery_payload)},
    )
    recovery_json = extract_json_from_mcp(recovery_result)

    print("\n=== 3) 恢复建议 ===")
    print(json.dumps(recovery_json, indent=2, ensure_ascii=False))

    await asyncio.sleep(0.2)
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
