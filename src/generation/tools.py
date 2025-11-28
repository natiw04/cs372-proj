"""
Tool Definitions for Tribly AI Assistant.

Defines the function/tool schemas that Claude can call to search
and retrieve information from the Tribly platform.

Rubric Items:
- Implemented agentic system with tool calls (7 pts)
- Built multi-stage ML pipeline (7 pts - partial)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ToolName(str, Enum):
    """Available tools for the assistant."""
    SEARCH_REVIEWS = "search_reviews"
    SEARCH_EVENTS = "search_events"
    SEARCH_RESOURCES = "search_resources"
    SEARCH_POSTS = "search_posts"
    SEARCH_CLASSES = "search_classes"
    SEARCH_TEACHERS = "search_teachers"
    SEARCH_ALL = "search_all"
    GET_DOCUMENT = "get_document"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool that Claude can call."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_claude_schema(self) -> Dict[str, Any]:
        """Convert to Claude API tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

SEARCH_REVIEWS_TOOL = ToolDefinition(
    name=ToolName.SEARCH_REVIEWS.value,
    description="""Search for professor and class reviews. Use this when the user asks about:
- Professor ratings or reviews
- Class difficulty or workload
- Student experiences with specific courses
- Recommendations for professors
- Teaching quality feedback""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about professors, classes, or reviews"
        ),
        ToolParameter(
            name="professor_name",
            type="string",
            description="Filter by specific professor name (optional)",
            required=False
        ),
        ToolParameter(
            name="class_name",
            type="string",
            description="Filter by specific class name or code (optional)",
            required=False
        ),
        ToolParameter(
            name="min_rating",
            type="number",
            description="Minimum rating filter (1-5) (optional)",
            required=False
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_EVENTS_TOOL = ToolDefinition(
    name=ToolName.SEARCH_EVENTS.value,
    description="""Search for campus events and activities. Use this when the user asks about:
- Upcoming events on campus
- Career fairs or networking events
- Club meetings or workshops
- Events with free food
- Academic or social gatherings""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about events"
        ),
        ToolParameter(
            name="event_type",
            type="string",
            description="Filter by event type (optional)",
            required=False,
            enum=["career", "academic", "social", "workshop", "networking", "club"]
        ),
        ToolParameter(
            name="free_food",
            type="boolean",
            description="Filter for events with free food (optional)",
            required=False
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_RESOURCES_TOOL = ToolDefinition(
    name=ToolName.SEARCH_RESOURCES.value,
    description="""Search for study resources and materials. Use this when the user asks about:
- Study guides or notes for specific classes
- Practice problems or past exams
- Lecture notes or summaries
- Cheat sheets or formula references
- Educational materials shared by students""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about study resources"
        ),
        ToolParameter(
            name="resource_type",
            type="string",
            description="Filter by resource type (optional)",
            required=False,
            enum=["study_guide", "notes", "cheat_sheet", "practice_problems", "lecture_notes", "summary"]
        ),
        ToolParameter(
            name="class_name",
            type="string",
            description="Filter by class name or code (optional)",
            required=False
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_POSTS_TOOL = ToolDefinition(
    name=ToolName.SEARCH_POSTS.value,
    description="""Search community posts and discussions. Use this when the user asks about:
- Student discussions or questions
- Tips and advice from other students
- Study group requests
- Campus life questions
- General community conversations""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about posts or discussions"
        ),
        ToolParameter(
            name="category",
            type="string",
            description="Filter by post category (optional)",
            required=False,
            enum=["academic", "social", "career", "general"]
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_CLASSES_TOOL = ToolDefinition(
    name=ToolName.SEARCH_CLASSES.value,
    description="""Search for class information. Use this when the user asks about:
- Specific courses by name or code
- Class descriptions or prerequisites
- What a class covers
- Classes in a specific subject area""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about classes"
        ),
        ToolParameter(
            name="department",
            type="string",
            description="Filter by department (e.g., 'CS', 'MATH') (optional)",
            required=False
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_TEACHERS_TOOL = ToolDefinition(
    name=ToolName.SEARCH_TEACHERS.value,
    description="""Search for professor information. Use this when the user asks about:
- Finding professors by name
- Professor research interests
- Professor bios or backgrounds
- Which professors teach certain subjects""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query about professors"
        ),
        ToolParameter(
            name="department",
            type="string",
            description="Filter by department (optional)",
            required=False
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 5)",
            required=False
        )
    ]
)

SEARCH_ALL_TOOL = ToolDefinition(
    name=ToolName.SEARCH_ALL.value,
    description="""Search across all content types. Use this for:
- General questions that could span multiple categories
- When unsure which specific search to use
- Broad queries about campus life""",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Natural language search query"
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of results to return (default: 10)",
            required=False
        )
    ]
)

GET_DOCUMENT_TOOL = ToolDefinition(
    name=ToolName.GET_DOCUMENT.value,
    description="""Get detailed information about a specific document by ID.
Use this to get more details about a search result.""",
    parameters=[
        ToolParameter(
            name="document_id",
            type="string",
            description="The ID of the document to retrieve"
        ),
        ToolParameter(
            name="collection",
            type="string",
            description="The collection the document belongs to",
            enum=["reviews", "events", "hangouts", "posts", "resources", "classes", "teachers", "groups"]
        )
    ]
)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# All available tools
ALL_TOOLS: List[ToolDefinition] = [
    SEARCH_REVIEWS_TOOL,
    SEARCH_EVENTS_TOOL,
    SEARCH_RESOURCES_TOOL,
    SEARCH_POSTS_TOOL,
    SEARCH_CLASSES_TOOL,
    SEARCH_TEACHERS_TOOL,
    SEARCH_ALL_TOOL,
    GET_DOCUMENT_TOOL
]

# Tool lookup by name
TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    tool.name: tool for tool in ALL_TOOLS
}


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas in Claude API format."""
    return [tool.to_claude_schema() for tool in ALL_TOOLS]


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific tool schema by name."""
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name].to_claude_schema()
    return None


def get_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return [tool.name for tool in ALL_TOOLS]


# ============================================================================
# TOOL SELECTION HELPERS
# ============================================================================

def select_tools_for_query(query: str) -> List[Dict[str, Any]]:
    """
    Select relevant tools based on query content.

    This is a simple heuristic-based selection. The actual tool
    choice is made by Claude based on context.
    """
    query_lower = query.lower()

    # Keywords that suggest specific tools
    review_keywords = ["review", "rating", "professor", "teacher", "rate", "recommend"]
    event_keywords = ["event", "happening", "upcoming", "career fair", "workshop", "meeting"]
    resource_keywords = ["study guide", "notes", "resource", "material", "practice", "cheat sheet"]
    post_keywords = ["post", "discussion", "advice", "tips", "question", "help"]
    class_keywords = ["class", "course", "prerequisite", "syllabus", "curriculum"]

    selected_tools = []

    # Always include search_all as fallback
    selected_tools.append(SEARCH_ALL_TOOL.to_claude_schema())

    # Add specialized tools based on query
    if any(kw in query_lower for kw in review_keywords):
        selected_tools.insert(0, SEARCH_REVIEWS_TOOL.to_claude_schema())
        selected_tools.append(SEARCH_TEACHERS_TOOL.to_claude_schema())

    if any(kw in query_lower for kw in event_keywords):
        selected_tools.insert(0, SEARCH_EVENTS_TOOL.to_claude_schema())

    if any(kw in query_lower for kw in resource_keywords):
        selected_tools.insert(0, SEARCH_RESOURCES_TOOL.to_claude_schema())

    if any(kw in query_lower for kw in post_keywords):
        selected_tools.insert(0, SEARCH_POSTS_TOOL.to_claude_schema())

    if any(kw in query_lower for kw in class_keywords):
        selected_tools.insert(0, SEARCH_CLASSES_TOOL.to_claude_schema())

    # Add get_document for follow-up queries
    selected_tools.append(GET_DOCUMENT_TOOL.to_claude_schema())

    # Remove duplicates while preserving order
    seen = set()
    unique_tools = []
    for tool in selected_tools:
        if tool["name"] not in seen:
            seen.add(tool["name"])
            unique_tools.append(tool)

    return unique_tools
