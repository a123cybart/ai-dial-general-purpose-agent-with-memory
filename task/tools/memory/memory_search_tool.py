import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        return "search_memory_tool"

    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "Tool for searching long-term memories about the user before every query the user makes to avoid asking the same questions repeatedly."

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema:
        #  - query is string, description: "The search query. Can be a question or keywords to find relevant memories", required
        #  - top_k is integer, description: "Number of most relevant memories to return.", minimum is 1, maximum is 20, default is 5
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        # TODO:
        # 1. Load arguments with `json`
        args = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `query` from arguments
        query = args["query"]
        # 3. Get `top_k` from arguments, default is 5
        top_k = args.get("top_k", 5)
        # 4. Call `memory_store` `search_memories` (we will implement logic in `memory_store` later)
        results = await self.memory_store.search_memories(
            query=query, top_k=top_k, api_key=tool_call_params.api_key
        )
        # 5. If results are empty then set `final_result` as "No memories found.",
        if not results:
            final_result = "No memories found."
        #    otherwise iterate through results and collect content, category and topics (if preset) in markdown format
        else:
            markdown_lines = []
            for memory in results:
                line = f"- **Content:** {memory.content}"
                line += f"\n  **Category:** {memory.category}"
                if memory.topics:
                    line += f"\n  **Topics:** {', '.join(memory.topics)}"
                markdown_lines.append(line)
            final_result = "\n".join(markdown_lines)
        # 6. Add result to stage as markdown text
        tool_call_params.stage.append_content(final_result)
        # 7. Return result
        return final_result
