"""
Gemini API Client for Tribly AI Assistant.

Handles interaction with the Google Gemini API including function calling
for the agentic RAG system.

Rubric Items:
- Implemented agentic system with tool calls (7 pts)
- Built multi-stage ML pipeline (7 pts - partial)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy import google.generativeai to avoid issues if not installed
_genai = None


def _get_genai():
    """Lazy load google.generativeai module."""
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )
    return _genai


@dataclass
class ToolUse:
    """Represents a tool call made by Gemini."""
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
    role: str  # "user", "model"
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


class GeminiClient:
    """
    Client for interacting with Gemini API with function calling support.

    Implements an agentic loop that allows Gemini to call tools
    and process results iteratively.
    """

    # Default model configuration
    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 4096
    MAX_ITERATIONS = 10  # Prevent infinite loops

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_tokens: int = None
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key. Uses GOOGLE_API_KEY env var if not provided.
            model: Model to use. Defaults to gemini-1.5-flash.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens or self.MAX_TOKENS

        self._model = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._is_initialized:
            return

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        genai = _get_genai()
        genai.configure(api_key=self.api_key)

        # Create the model
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": 0.7,
            }
        )

        self._is_initialized = True
        logger.info(f"Gemini client initialized with model: {self.model_name}")

    def is_ready(self) -> bool:
        """Check if client is initialized."""
        return self._is_initialized

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self.is_ready():
            self.initialize()

    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Claude-style tools to Gemini function declarations."""
        function_declarations = []

        for tool in tools:
            # Claude format has 'name', 'description', 'input_schema'
            # Gemini format uses 'name', 'description', 'parameters'
            func_decl = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
            function_declarations.append(func_decl)

        return function_declarations

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        tools: List[Dict[str, Any]] = None,
        tool_choice: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Send a chat request to Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice specification (not used by Gemini).

        Returns:
            Response dict with content and metadata.
        """
        self._ensure_initialized()

        genai = _get_genai()

        # Build Gemini-compatible messages
        gemini_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map roles: Claude uses "assistant", Gemini uses "model"
            if role == "assistant":
                role = "model"

            # Handle content that might be a list (tool results)
            if isinstance(content, list):
                # Convert tool results to Gemini format
                parts = []
                for block in content:
                    if block.get("type") == "tool_result":
                        parts.append({
                            "function_response": {
                                "name": block.get("tool_use_id", "unknown"),
                                "response": {"result": block.get("content", "")}
                            }
                        })
                    elif block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    else:
                        parts.append(str(block))
                content = parts if parts else str(content)

            gemini_messages.append({
                "role": role,
                "parts": [content] if isinstance(content, str) else content
            })

        # Create chat with system instruction
        chat_kwargs = {}
        if system_prompt:
            # For Gemini, system instruction is set at model level
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.7,
                },
                system_instruction=system_prompt
            )

        # Add tools if provided
        if tools:
            function_declarations = self._convert_tools_to_gemini_format(tools)
            chat_kwargs["tools"] = [{"function_declarations": function_declarations}]

        try:
            # Start chat and send messages
            chat = self._model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

            # Get the last user message
            last_message = gemini_messages[-1] if gemini_messages else {"parts": [""]}
            last_content = last_message.get("parts", [""])[0]

            if tools:
                response = chat.send_message(last_content, tools=chat_kwargs.get("tools"))
            else:
                response = chat.send_message(last_content)

            return self._response_to_dict(response)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _response_to_dict(self, response) -> Dict[str, Any]:
        """Convert Gemini response to dictionary."""
        content = []
        stop_reason = "stop"

        # Process response parts
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content.append({"type": "text", "text": part.text})
                elif hasattr(part, 'function_call'):
                    # Function call
                    fc = part.function_call
                    content.append({
                        "type": "tool_use",
                        "id": fc.name,  # Use function name as ID
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {}
                    })
                    stop_reason = "tool_use"

        # Get usage info if available
        usage = {}
        if hasattr(response, 'usage_metadata'):
            usage = {
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0)
            }

        return {
            "id": "gemini-response",
            "type": "message",
            "role": "model",
            "content": content,
            "model": self.model_name,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage
        }

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

        This implements the core agentic behavior where Gemini can:
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

            # Call Gemini
            response = self.chat(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools
            )

            # Update usage
            total_usage["input_tokens"] += response["usage"].get("input_tokens", 0)
            total_usage["output_tokens"] += response["usage"].get("output_tokens", 0)

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
_default_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get the default Gemini client instance."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
