"""
Prompt Templates for Tribly AI Assistant.

Contains system prompts and templates for various assistant behaviors.

Rubric Items:
- Built RAG system with document retrieval (10 pts - partial)
- Implemented agentic system with tool calls (7 pts - partial)
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

ASSISTANT_SYSTEM_PROMPT = """You are Tribly Assistant, an AI helper for university students using the Tribly platform.

## Your Role
You help students navigate university life by finding relevant information about:
- Professor and class reviews
- Campus events and activities
- Study resources and materials
- Community discussions and posts
- Student hangouts and social activities

## Guidelines

### Using Tools
- Use the provided search tools to find relevant information before responding
- Search for specific information rather than making up answers
- If a search returns no results, try a broader or different query
- You can use multiple tools to gather comprehensive information

### Response Style
- Be helpful, friendly, and concise
- Cite specific results when referencing information from searches
- If information might be outdated, mention that to the user
- When showing reviews, include ratings and key points
- For events, include date, time, and location when available

### Limitations
- You can only access information indexed in the Tribly platform
- You cannot create, modify, or delete any data
- You cannot access real-time information or external websites
- If you don't find relevant information, honestly say so

### Privacy
- Never reveal personal information about users
- Respect anonymous reviews and posts
- Don't try to identify anonymous authors

## Available Information Types
1. **Reviews**: Student reviews of professors and classes (ratings, comments, helpfulness)
2. **Events**: Campus events, career fairs, workshops, club meetings
3. **Resources**: Study guides, notes, practice problems, lecture materials
4. **Posts**: Student discussions, questions, tips, advice
5. **Classes**: Course information, descriptions, ratings
6. **Teachers**: Professor profiles, research interests, ratings
7. **Groups**: Classes and student organizations

When you receive a query, think about which type(s) of information would be most helpful and use the appropriate search tool(s)."""


# Shorter version for cost-sensitive usage
ASSISTANT_SYSTEM_PROMPT_COMPACT = """You are Tribly Assistant, helping university students find information about professors, classes, events, resources, and campus life.

Use the search tools to find relevant information. Be helpful and concise. Cite specific results. If no results found, suggest alternative searches.

Available searches: reviews, events, resources, posts, classes, teachers, or search_all."""


# ============================================================================
# QUERY ENHANCEMENT PROMPTS
# ============================================================================

QUERY_ENHANCEMENT_PROMPT = """Given the user's question, generate an improved search query that will help find relevant results in a semantic search system.

User question: {query}

Guidelines:
- Extract key concepts and entities
- Expand abbreviations (CS -> Computer Science, ML -> Machine Learning)
- Include relevant synonyms
- Remove filler words
- Keep it concise but comprehensive

Enhanced query:"""


# ============================================================================
# RESPONSE FORMATTING TEMPLATES
# ============================================================================

REVIEW_RESULT_TEMPLATE = """
**Review** (Rating: {rating}/5)
{comment}
{helpful_info}
"""

EVENT_RESULT_TEMPLATE = """
**{title}**
- Date: {date_time}
- Location: {location}
{free_food_info}
{description}
"""

RESOURCE_RESULT_TEMPLATE = """
**{title}** ({resource_type})
{description}
Upvotes: {upvotes}
"""

POST_RESULT_TEMPLATE = """
**{title}**
Category: {category} | Upvotes: {upvotes} | Comments: {comment_count}
{content}
"""


# ============================================================================
# SPECIALIZED PROMPTS
# ============================================================================

REVIEW_ANALYSIS_PROMPT = """Based on the following reviews, provide a summary analysis:

{reviews}

Please summarize:
1. Overall sentiment (positive/mixed/negative)
2. Common themes or points mentioned
3. Key strengths mentioned
4. Key concerns or criticisms
5. A brief recommendation"""


EVENT_RECOMMENDATION_PROMPT = """Based on the user's interests and the following events, recommend the most relevant ones:

User interests: {interests}

Available events:
{events}

Provide personalized recommendations with brief explanations of why each event might interest them."""


STUDY_RESOURCE_PROMPT = """Help the student find study resources for their class:

Class: {class_name}
Topic/Need: {topic}

Available resources:
{resources}

Recommend the most helpful resources and explain how they can use them effectively."""


# ============================================================================
# ERROR AND FALLBACK MESSAGES
# ============================================================================

NO_RESULTS_MESSAGE = """I couldn't find any results for your query. Here are some suggestions:

1. Try using different keywords or broader terms
2. Check for any typos in your search
3. If searching for a specific class, try the course code (e.g., "CS201")
4. For professors, try searching by last name only

Would you like me to try a different search?"""


API_ERROR_MESSAGE = """I'm having trouble accessing the search system right now. Please try again in a moment. If the problem persists, you can try:

1. Refreshing the conversation
2. Simplifying your question
3. Trying again later"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_review_result(result: Dict[str, Any]) -> str:
    """Format a single review result."""
    rating = result.get("rating", "N/A")
    comment = result.get("content", result.get("comment", "No comment available"))
    helpful = result.get("helpful_count", 0)

    helpful_info = f"({helpful} students found this helpful)" if helpful > 0 else ""

    return REVIEW_RESULT_TEMPLATE.format(
        rating=rating,
        comment=comment,
        helpful_info=helpful_info
    ).strip()


def format_event_result(result: Dict[str, Any]) -> str:
    """Format a single event result."""
    title = result.get("title", "Untitled Event")
    date_time = result.get("date_time", "TBD")
    location = result.get("location", "TBD")
    free_food = result.get("free_food", False)
    description = result.get("content", result.get("description", ""))[:200]

    free_food_info = "Free food available!" if free_food else ""

    return EVENT_RESULT_TEMPLATE.format(
        title=title,
        date_time=date_time,
        location=location,
        free_food_info=free_food_info,
        description=description
    ).strip()


def format_resource_result(result: Dict[str, Any]) -> str:
    """Format a single resource result."""
    title = result.get("title", "Untitled Resource")
    resource_type = result.get("resource_type", result.get("type", "resource"))
    description = result.get("content", result.get("description", ""))[:200]
    upvotes = result.get("upvotes", 0)

    return RESOURCE_RESULT_TEMPLATE.format(
        title=title,
        resource_type=resource_type.replace("_", " ").title(),
        description=description,
        upvotes=upvotes
    ).strip()


def format_post_result(result: Dict[str, Any]) -> str:
    """Format a single post result."""
    title = result.get("title", "Untitled Post")
    category = result.get("category", "general")
    upvotes = result.get("upvotes", 0)
    comment_count = result.get("comment_count", 0)
    content = result.get("content", "")[:300]

    return POST_RESULT_TEMPLATE.format(
        title=title,
        category=category,
        upvotes=upvotes,
        comment_count=comment_count,
        content=content
    ).strip()


def format_search_results(
    results: List[Dict[str, Any]],
    result_type: str = "mixed"
) -> str:
    """Format a list of search results."""
    if not results:
        return NO_RESULTS_MESSAGE

    formatters = {
        "review": format_review_result,
        "event": format_event_result,
        "resource": format_resource_result,
        "post": format_post_result
    }

    formatted_parts = []
    for i, result in enumerate(results, 1):
        r_type = result.get("type", result_type)
        formatter = formatters.get(r_type, lambda x: str(x))

        try:
            formatted = formatter(result)
            formatted_parts.append(f"{i}. {formatted}")
        except Exception:
            formatted_parts.append(f"{i}. {result.get('content', 'Result unavailable')[:200]}")

    return "\n\n".join(formatted_parts)


def build_context_prompt(
    user_query: str,
    search_results: List[Dict[str, Any]],
    include_system: bool = True
) -> str:
    """
    Build a complete prompt with search results as context.

    This is useful for simple RAG without tool calling.
    """
    context = format_search_results(search_results)

    prompt = f"""Based on the following search results, answer the user's question.

## Search Results
{context}

## User Question
{user_query}

Please provide a helpful response based on the search results above. If the results don't contain relevant information, say so and suggest how the user might find what they're looking for."""

    if include_system:
        return f"{ASSISTANT_SYSTEM_PROMPT_COMPACT}\n\n{prompt}"

    return prompt
