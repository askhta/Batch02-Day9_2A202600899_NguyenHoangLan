"""Bài Tập 2: Thêm Tools và Knowledge Base"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "tranh chấp lao động", "bộ luật lao động"],
        "text": (
            "Theo pháp luật lao động Việt Nam, tranh chấp lao động liên quan đến sa thải, "
            "đơn phương chấm dứt hợp đồng lao động, tiền lương, bồi thường, bảo hiểm xã hội "
            "cần được xem xét theo Bộ luật Lao động và các văn bản hướng dẫn liên quan."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ việc."""
    case_lower = case_type.lower()

    if any(kw in case_lower for kw in ["contract", "hợp đồng", "vi phạm hợp đồng"]):
        return "Thời hiệu khởi kiện tranh chấp hợp đồng thường là 03 năm kể từ ngày người có quyền biết hoặc phải biết quyền lợi của mình bị xâm phạm."

    if any(kw in case_lower for kw in ["lao động", "sa thải", "hợp đồng lao động"]):
        return "Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân thường là 01 năm kể từ ngày phát hiện quyền, lợi ích hợp pháp bị vi phạm."

    if any(kw in case_lower for kw in ["tort", "bồi thường", "thiệt hại", "ngoài hợp đồng"]):
        return "Thời hiệu khởi kiện yêu cầu bồi thường thiệt hại ngoài hợp đồng thường là 03 năm."

    return "Chưa xác định được loại vụ việc. Vui lòng nêu rõ: hợp đồng, lao động, bồi thường thiệt hại..."


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)

    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"

    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]

    print(f"Câu hỏi: {question}\n")

    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None

            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])

            if tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])

            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\nKết quả:\n{final_response.content}")
    else:
        print(f"\nKết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())