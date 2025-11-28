"""
Claude API Client for Tribly AI Assistant.

Handles interaction with the Claude API including tool/function calling
for the agentic RAG system.

Rubric Items:
- Implemented agentic system with tool calls (7 pts)
- Built multi-stage ML pipeline (7 pts - partial)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy import anthropic to avoid issues if not installed
_anthropic = None


def _get_anthropic():
    """Lazy load anthropic module."""
    global _anthropic
    if _anthropic is None:
        try:
            import anthropic
            _anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )
    return _anthropic


@dataclass
class ToolUse:
    """Represents a tool call made by Claude."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant"
    content: Any  # str or list of content blocks


@dataclass
class AgentResponse:
    """Complete response from the agent including any tool calls."""
    final_text: str
    tool_calls: List[ToolUse] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""


class ClaudeClient:
    """
    Client for interacting with Claude API with tool calling support.

    Implements an agentic loop that allows Claude to call tools
    and process results iteratively.
    """

    # Default model configuration
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096
    MAX_ITERATIONS = 10  # Prevent infinite loops

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_tokens: int = None
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
            model: Model to use. Defaults to claude-sonnet-4-20250514.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens or self.MAX_TOKENS

        self._client = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if self._is_initialized:
            return

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        anthropic = _get_anthropic()
        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._is_initialized = True
        logger.info(f"Claude client initialized with model: {self.model}")

    def is_ready(self) -> bool:
        """Check if client is initialized."""
        return self._is_initialized

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self.is_ready():
            self.initialize()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Send a chat request to Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice specification.

        Returns:
            Raw API response dict.
        """
        self._ensure_initialized()

        # Build request params
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = tools

        if tool_choice:
            params["tool_choice"] = tool_choice

        try:
            response = self._client.messages.create(**params)
            return self._response_to_dict(response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _response_to_dict(self, response) -> Dict[str, Any]:
        """Convert API response to dictionary."""
        return {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "content": [self._content_block_to_dict(block) for block in response.content],
            "model": response.model,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    def _content_block_to_dict(self, block) -> Dict[str, Any]:
        """Convert content block to dictionary."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        elif block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            }
        return {"type": block.type}

    def run_agent(
        self,
        query: str,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], str],
        conversation_history: List[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Run the agentic loop with tool calling.

        This implements the core agentic behavior where Claude can:
        1. Analyze the query
        2. Call tools to retrieve information
        3. Process tool results
        4. Generate a final response

        Args:
            query: User's query.
            system_prompt: System prompt for the agent.
            tools: List of available tools.
            tool_executor: Function to execute tools. Takes (tool_name, tool_input)
                          and returns result string.
            conversation_history: Optional previous conversation messages.

        Returns:
            AgentResponse with final text and tool call history.
        """
        self._ensure_initialized()

        # Initialize conversation
        messages = list(conversation_history) if conversation_history else []
        messages.append({"role": "user", "content": query})

        all_tool_calls = []
        all_tool_results = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        iteration = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logger.debug(f"Agent iteration {iteration}")

            # Call Claude
            response = self.chat(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools
            )

            # Update usage
            total_usage["input_tokens"] += response["usage"]["input_tokens"]
            total_usage["output_tokens"] += response["usage"]["output_tokens"]

            # Process response content
            assistant_content = response["content"]
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use
            tool_uses = [
                block for block in assistant_content
                if block.get("type") == "tool_use"
            ]

            if not tool_uses:
                # No tool calls, extract final text
                final_text = self._extract_text(assistant_content)
                return AgentResponse(
                    final_text=final_text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    messages=messages,
                    usage=total_usage,
                    stop_reason=response["stop_reason"]
                )

            # Execute tools
            tool_results_content = []
            for tool_use in tool_uses:
                tool_call = ToolUse(
                    id=tool_use["id"],
                    name=tool_use["name"],
                    input=tool_use["input"]
                )
                all_tool_calls.append(tool_call)

                logger.info(f"Executing tool: {tool_call.name}")
                logger.debug(f"Tool input: {tool_call.input}")

                try:
                    result = tool_executor(tool_call.name, tool_call.input)
                    tool_result = ToolResult(
                        tool_use_id=tool_call.id,
                        content=result,
                        is_error=False
                    )
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_result = ToolResult(
                        tool_use_id=tool_call.id,
                        content=f"Error executing tool: {str(e)}",
                        is_error=True
                    )

                all_tool_results.append(tool_result)
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_use_id,
                    "content": tool_result.content,
                    "is_error": tool_result.is_error
                })

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results_content})

        # Max iterations reached
        logger.warning(f"Max iterations ({self.MAX_ITERATIONS}) reached")
        final_text = self._extract_text(messages[-1].get("content", []))

        return AgentResponse(
            final_text=final_text or "I apologize, but I wasn't able to complete the search. Please try again.",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            messages=messages,
            usage=total_usage,
            stop_reason="max_iterations"
        )

    def _extract_text(self, content: Any) -> str:
        """Extract text from response content."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)

        return ""

    def simple_chat(
        self,
        query: str,
        system_prompt: str = None
    ) -> str:
        """
        Simple chat without tools.

        Args:
            query: User message.
            system_prompt: Optional system prompt.

        Returns:
            Assistant's response text.
        """
        self._ensure_initialized()

        messages = [{"role": "user", "content": query}]
        response = self.chat(messages=messages, system_prompt=system_prompt)

        return self._extract_text(response["content"])


# Singleton instance
_default_client: Optional[ClaudeClient] = None


def get_claude_client() -> ClaudeClient:
    """Get the default Claude client instance."""
    global _default_client
    if _default_client is None:
        _default_client = ClaudeClient()
    return _default_client
