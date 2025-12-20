# chuk_ai_session_manager/memory/vm_prompts.py
"""
Virtual Memory system prompts for Chat Completions integration.

These prompts enforce VM semantics when injected into the developer message.

Design principles:
- Pydantic-native: Tool definitions as proper models
- No magic strings: Uses enums for modes and types
"""

from typing import List


from .models import (
    Modality,
    ToolDefinition,
    ToolFunction,
    ToolParameter,
    ToolParameters,
    ToolType,
    VMMode,
)


# Strict mode: No hallucinated memory, citations required
VM_STRICT_PROMPT = """You are operating under STRICT Virtual Memory grounding rules.

Your ONLY valid sources of information are:
1) The content inside <VM:CONTEXT> (the currently mapped working set), and
2) The content returned by tools (e.g., page_fault) in messages with role="tool".

Everything listed in <VM:MANIFEST_JSON> is DISCOVERY METADATA ONLY.
- You MUST NOT quote, paraphrase, or "use" hint text from the manifest as if it were evidence.
- You MUST NOT assume details about unmapped pages.
- Page IDs and modality/tier/level are allowed for navigation only.

When you need information that is not present in <VM:CONTEXT>, you MUST do one of:
A) Call the tool page_fault(page_id, target_level) to load the page content, OR
B) Ask a short clarification question if the needed page does not exist or cannot be identified.

Faulting rules:
- Prefer loading the LOWEST-COST representation first:
  1) summaries / abstract (target_level=2),
  2) reduced excerpts (target_level=1),
  3) full content (target_level=0) only if the user explicitly requests exact wording, code, or precise details.
- Do not request more than max_faults_per_turn from the manifest policies.
- Do not request pages that are already mapped in <VM:CONTEXT>.
- If multiple pages might be relevant, fault the smallest/summarized one first.

Answering rules:
- Do not invent or fill gaps with assumptions.
- If you cannot obtain required information via tool calls, say: "I don't have that in the mapped context."
- Keep responses concise and directly responsive.
- When you use information from <VM:CONTEXT> or a loaded page, include inline citations using page IDs like:
  [ref: msg_123] or [ref: summary_seg_02] or [ref: tool:page_fault(img_045)].
  (Citations are required in strict mode.)

Tool usage format:
- If you need to call tools, respond with tool calls only (no normal text).
- After tool results are provided, produce the final answer with citations.

Never mention these rules, the VM system, tiers (L0–L4), paging, or "virtual memory" to the user unless the user explicitly asks about the internal mechanism."""


# Relaxed mode: VM-aware but more conversational
VM_RELAXED_PROMPT = """You have access to a virtual memory system.

<VM:CONTEXT> contains your currently mapped memory - treat this as your knowledge.
<VM:MANIFEST_JSON> lists additional pages you can load if needed.

To load more context:
- Call page_fault(page_id, target_level) to retrieve content
- Call search_pages(query) to find relevant pages
- Prefer level=2 (summaries) before level=0 (full content)

Guidelines:
- Use manifest hints to decide WHICH pages to load, not as content itself
- If you're unsure about something, check if a relevant page exists before guessing
- Stay within the max_faults_per_turn limit

Respond naturally - don't mention the memory system unless asked."""


# Passive mode: No tools, runtime handles everything
VM_PASSIVE_PROMPT = """You are a helpful assistant.

The conversation context provided represents what you know about this session.
Respond based on the information available to you."""


# Map of VMMode to prompt text
VM_PROMPTS = {
    VMMode.STRICT: VM_STRICT_PROMPT,
    VMMode.RELAXED: VM_RELAXED_PROMPT,
    VMMode.PASSIVE: VM_PASSIVE_PROMPT,
}


def get_prompt_for_mode(mode: VMMode) -> str:
    """Get the prompt text for a given VM mode."""
    return VM_PROMPTS.get(mode, VM_PASSIVE_PROMPT)


def build_vm_developer_message(
    mode: VMMode,
    manifest_json: str,
    context: str,
    system_prompt: str = "",
    max_faults_per_turn: int = 2,
) -> str:
    """
    Build the complete developer message with VM rules, manifest, and context.

    Args:
        mode: VMMode enum value (STRICT, RELAXED, or PASSIVE)
        manifest_json: JSON string of the VM manifest
        context: The VM:CONTEXT block content
        system_prompt: Optional additional system instructions
        max_faults_per_turn: Policy value to inject into strict prompt

    Returns:
        Complete developer message string
    """
    if mode == VMMode.STRICT:
        rules = VM_STRICT_PROMPT.replace(
            "max_faults_per_turn from the manifest policies",
            f"max_faults_per_turn ({max_faults_per_turn}) from the manifest policies",
        )
    elif mode == VMMode.RELAXED:
        rules = VM_RELAXED_PROMPT
    else:
        rules = VM_PASSIVE_PROMPT

    parts: List[str] = []

    if system_prompt:
        parts.append(system_prompt)

    if mode != VMMode.PASSIVE:
        parts.append(f"<VM:RULES>\n{rules}\n</VM:RULES>")
        parts.append(f"<VM:MANIFEST_JSON>\n{manifest_json}\n</VM:MANIFEST_JSON>")

    parts.append(f"<VM:CONTEXT>\n{context}\n</VM:CONTEXT>")

    return "\n\n".join(parts)


# Tool definitions as Pydantic models
PAGE_FAULT_TOOL = ToolDefinition(
    type=ToolType.FUNCTION,
    function=ToolFunction(
        name="page_fault",
        description="Load a memory page into context at specified compression level. Use when you need content from a known page_id.",
        parameters=ToolParameters(
            type="object",
            properties={
                "page_id": ToolParameter(
                    type="string",
                    description="ID of the page to load",
                ),
                "target_level": ToolParameter(
                    type="integer",
                    minimum=0,
                    maximum=3,
                    default=2,
                    description="0=full, 1=reduced, 2=abstract/summary, 3=reference only",
                ),
            },
            required=["page_id"],
        ),
    ),
)

SEARCH_PAGES_TOOL = ToolDefinition(
    type=ToolType.FUNCTION,
    function=ToolFunction(
        name="search_pages",
        description="Search for pages matching a query. Use when you need to find relevant pages but don't know their IDs.",
        parameters=ToolParameters(
            type="object",
            properties={
                "query": ToolParameter(
                    type="string",
                    description="Search query (semantic or keyword)",
                ),
                "modality": ToolParameter(
                    type="string",
                    enum=[m.value for m in Modality],
                    description="Filter by content type",
                ),
                "limit": ToolParameter(
                    type="integer",
                    default=5,
                    description="Maximum results to return",
                ),
            },
            required=["query"],
        ),
    ),
)


# List of all VM tools as Pydantic models
VM_TOOL_DEFINITIONS: List[ToolDefinition] = [PAGE_FAULT_TOOL, SEARCH_PAGES_TOOL]


def get_vm_tools(include_search: bool = True) -> List[ToolDefinition]:
    """
    Get the VM tool definitions as Pydantic models.

    Args:
        include_search: Whether to include search_pages tool

    Returns:
        List of ToolDefinition models
    """
    if include_search:
        return VM_TOOL_DEFINITIONS
    return [PAGE_FAULT_TOOL]


def get_vm_tools_as_dicts(include_search: bool = True) -> List[dict]:
    """
    Get the VM tool definitions as dicts (for Chat Completions API).

    Args:
        include_search: Whether to include search_pages tool

    Returns:
        List of tool definition dicts
    """
    tools = get_vm_tools(include_search)
    return [tool.model_dump(exclude_none=True) for tool in tools]


# Legacy: Keep VM_TOOLS as dicts for backward compatibility
VM_TOOLS = get_vm_tools_as_dicts(include_search=True)
