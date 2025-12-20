# chuk_ai_session_manager/memory/vm_prompts.py
"""
Virtual Memory system prompts for Chat Completions integration.

These prompts enforce VM semantics when injected into the developer message.
"""

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


def build_vm_developer_message(
    mode: str,
    manifest_json: str,
    context: str,
    system_prompt: str = "",
    max_faults_per_turn: int = 2,
) -> str:
    """
    Build the complete developer message with VM rules, manifest, and context.

    Args:
        mode: "strict", "relaxed", or "passive"
        manifest_json: JSON string of the VM manifest
        context: The VM:CONTEXT block content
        system_prompt: Optional additional system instructions
        max_faults_per_turn: Policy value to inject into strict prompt

    Returns:
        Complete developer message string
    """
    if mode == "strict":
        rules = VM_STRICT_PROMPT.replace(
            "max_faults_per_turn from the manifest policies",
            f"max_faults_per_turn ({max_faults_per_turn}) from the manifest policies"
        )
    elif mode == "relaxed":
        rules = VM_RELAXED_PROMPT
    else:
        rules = VM_PASSIVE_PROMPT

    parts = []

    if system_prompt:
        parts.append(system_prompt)

    if mode != "passive":
        parts.append(f"<VM:RULES>\n{rules}\n</VM:RULES>")
        parts.append(f"<VM:MANIFEST_JSON>\n{manifest_json}\n</VM:MANIFEST_JSON>")

    parts.append(f"<VM:CONTEXT>\n{context}\n</VM:CONTEXT>")

    return "\n\n".join(parts)


# Tool definitions for VM syscalls
VM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "page_fault",
            "description": "Load a memory page into context at specified compression level. Use when you need content from a known page_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "ID of the page to load"
                    },
                    "target_level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                        "default": 2,
                        "description": "0=full, 1=reduced, 2=abstract/summary, 3=reference only"
                    }
                },
                "required": ["page_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pages",
            "description": "Search for pages matching a query. Use when you need to find relevant pages but don't know their IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (semantic or keyword)"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "image", "audio", "video", "structured"],
                        "description": "Filter by content type"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def get_vm_tools(include_search: bool = True) -> list:
    """
    Get the VM tool definitions for Chat Completions.

    Args:
        include_search: Whether to include search_pages tool

    Returns:
        List of tool definitions
    """
    if include_search:
        return VM_TOOLS
    return [VM_TOOLS[0]]  # Just page_fault
