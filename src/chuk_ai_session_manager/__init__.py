# chuk_ai_session_manager/__init__.py
"""
chuk_ai_session_manager - AI Session Management with CHUK Sessions Backend

A comprehensive session management system for AI conversations with support for:
- Hierarchical session organization with parent-child relationships
- Token usage tracking and cost estimation
- Tool execution logging and result caching
- Async-first design with proper error handling
- Multiple prompt building strategies for different use cases
- Infinite conversation support with automatic summarization
- Simple developer API for easy integration

Basic Usage:
    from chuk_ai_session_manager import SessionManager, Session, EventSource
    
    # Simple API
    sm = SessionManager()
    await sm.user_says("Hello!")
    await sm.ai_responds("Hi there!", model="gpt-4")
    
    # Advanced API
    session = await Session.create()
    event = await SessionEvent.create_with_tokens(
        message="Hello world",
        prompt="Hello world",
        model="gpt-4",
        source=EventSource.USER
    )
    await session.add_event_and_save(event)
"""

__version__ = "0.1.0"

# Import order is important to avoid circular dependencies
_imported_modules = {}

# Core enums (no dependencies)
try:
    from chuk_ai_session_manager.models.event_source import EventSource
    from chuk_ai_session_manager.models.event_type import EventType
    _imported_modules['enums'] = True
except ImportError as e:
    _imported_modules['enums'] = f"Import error: {e}"
    EventSource = None
    EventType = None

# Token usage (minimal dependencies)
try:
    from chuk_ai_session_manager.models.token_usage import TokenUsage, TokenSummary
    _imported_modules['token_usage'] = True
except ImportError as e:
    _imported_modules['token_usage'] = f"Import error: {e}"
    TokenUsage = None
    TokenSummary = None

# Session metadata (depends on nothing)
try:
    from chuk_ai_session_manager.models.session_metadata import SessionMetadata
    _imported_modules['session_metadata'] = True
except ImportError as e:
    _imported_modules['session_metadata'] = f"Import error: {e}"
    SessionMetadata = None

# Session run (minimal dependencies)
try:
    from chuk_ai_session_manager.models.session_run import SessionRun, RunStatus
    _imported_modules['session_run'] = True
except ImportError as e:
    _imported_modules['session_run'] = f"Import error: {e}"
    SessionRun = None
    RunStatus = None

# Session event (depends on enums and token usage)
try:
    from chuk_ai_session_manager.models.session_event import SessionEvent
    _imported_modules['session_event'] = True
except ImportError as e:
    _imported_modules['session_event'] = f"Import error: {e}"
    SessionEvent = None

# Session (depends on most other models)
try:
    from chuk_ai_session_manager.models.session import Session
    _imported_modules['session'] = True
except ImportError as e:
    _imported_modules['session'] = f"Import error: {e}"
    Session = None

# Storage backend
try:
    from chuk_ai_session_manager.session_storage import (
        SessionStorage, 
        get_backend, 
        setup_chuk_sessions_storage,
        ChukSessionsStore
    )
    from chuk_sessions import SessionManager as ChukSessionManager
    _imported_modules['storage'] = True
    
    # Create a compatibility alias for tests that expect chuk_sessions_storage
    import types
    chuk_sessions_storage = types.ModuleType('chuk_sessions_storage')
    chuk_sessions_storage.SessionStorage = SessionStorage
    chuk_sessions_storage.get_backend = get_backend
    chuk_sessions_storage.setup_chuk_sessions_storage = setup_chuk_sessions_storage
    chuk_sessions_storage.ChukSessionsStore = ChukSessionsStore
    chuk_sessions_storage.ChukSessionManager = ChukSessionManager
    
except ImportError as e:
    _imported_modules['storage'] = f"Import error: {e}"
    SessionStorage = None
    get_backend = None
    setup_chuk_sessions_storage = None
    ChukSessionsStore = None
    chuk_sessions_storage = None
    ChukSessionManager = None

# Exceptions
try:
    from chuk_ai_session_manager.exceptions import (
        SessionManagerError,
        SessionNotFound,
        SessionAlreadyExists,
        InvalidSessionOperation,
        TokenLimitExceeded,
        StorageError,
        ToolProcessingError
    )
    _imported_modules['exceptions'] = True
except ImportError as e:
    _imported_modules['exceptions'] = f"Import error: {e}"
    SessionManagerError = None
    SessionNotFound = None
    SessionAlreadyExists = None
    InvalidSessionOperation = None
    TokenLimitExceeded = None
    StorageError = None
    ToolProcessingError = None

# Prompt building
try:
    from chuk_ai_session_manager.session_prompt_builder import (
        PromptStrategy,
        build_prompt_from_session,
        truncate_prompt_to_token_limit
    )
    _imported_modules['prompt_builder'] = True
except ImportError as e:
    _imported_modules['prompt_builder'] = f"Import error: {e}"
    PromptStrategy = None
    build_prompt_from_session = None
    truncate_prompt_to_token_limit = None

# Tool processing
try:
    from chuk_ai_session_manager.session_aware_tool_processor import SessionAwareToolProcessor
    _imported_modules['tool_processor'] = True
except ImportError as e:
    _imported_modules['tool_processor'] = f"Import error: {e}"
    SessionAwareToolProcessor = None

# Infinite conversation management
try:
    from chuk_ai_session_manager.infinite_conversation import (
        InfiniteConversationManager,
        SummarizationStrategy
    )
    _imported_modules['infinite_conversation'] = True
except ImportError as e:
    _imported_modules['infinite_conversation'] = f"Import error: {e}"
    InfiniteConversationManager = None
    SummarizationStrategy = None

# Simple API (depends on most other components)
try:
    from chuk_ai_session_manager.simple_api import (
        SessionManager,
        quick_conversation,
        track_llm_call
    )
    _imported_modules['simple_api'] = True
except ImportError as e:
    _imported_modules['simple_api'] = f"Import error: {e}"
    SessionManager = None
    quick_conversation = None
    track_llm_call = None

# Build __all__ list from successfully imported components
__all__ = []

# Core models
if EventSource is not None:
    __all__.extend(['EventSource', 'EventType'])
if TokenUsage is not None:
    __all__.extend(['TokenUsage', 'TokenSummary'])
if SessionMetadata is not None:
    __all__.append('SessionMetadata')
if SessionRun is not None:
    __all__.extend(['SessionRun', 'RunStatus'])
if SessionEvent is not None:
    __all__.append('SessionEvent')
if Session is not None:
    __all__.append('Session')

# Storage
if SessionStorage is not None:
    __all__.extend([
        'SessionStorage', 
        'get_backend', 
        'setup_chuk_sessions_storage',
        'ChukSessionsStore',
        'ChukSessionManager',
        'chuk_sessions_storage'  # Add the compatibility alias
    ])

# Exceptions
if SessionManagerError is not None:
    __all__.extend([
        'SessionManagerError',
        'SessionNotFound',
        'SessionAlreadyExists',
        'InvalidSessionOperation',
        'TokenLimitExceeded',
        'StorageError',
        'ToolProcessingError'
    ])

# Advanced features
if PromptStrategy is not None:
    __all__.extend([
        'PromptStrategy',
        'build_prompt_from_session',
        'truncate_prompt_to_token_limit'
    ])

if SessionAwareToolProcessor is not None:
    __all__.append('SessionAwareToolProcessor')

if InfiniteConversationManager is not None:
    __all__.extend(['InfiniteConversationManager', 'SummarizationStrategy'])

# Simple API
if SessionManager is not None:
    __all__.extend(['SessionManager', 'quick_conversation', 'track_llm_call'])

def get_import_status():
    """
    Get the import status of all modules.
    
    Returns:
        Dict[str, Union[bool, str]]: Module import status
    """
    return _imported_modules.copy()

def check_dependencies():
    """
    Check which optional dependencies are available.
    
    Returns:
        Dict[str, bool]: Availability of optional dependencies
    """
    deps = {}
    
    # Check tiktoken for accurate token counting
    try:
        import tiktoken
        deps['tiktoken'] = True
    except ImportError:
        deps['tiktoken'] = False
    
    # Check chuk_sessions
    try:
        import chuk_sessions
        deps['chuk_sessions'] = True
    except ImportError:
        deps['chuk_sessions'] = False
    
    # Check chuk_tool_processor
    try:
        import chuk_tool_processor
        deps['chuk_tool_processor'] = True
    except ImportError:
        deps['chuk_tool_processor'] = False
    
    return deps

# Convenience function to check if package is properly set up
def verify_installation():
    """
    Verify that the package is properly installed and configured.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []
    
    # Check core components
    if Session is None:
        issues.append("Core Session model failed to import")
    if SessionEvent is None:
        issues.append("SessionEvent model failed to import")
    if SessionManager is None:
        issues.append("Simple API (SessionManager) failed to import")
    
    # Check dependencies
    deps = check_dependencies()
    if not deps.get('chuk_sessions', False):
        issues.append("chuk_sessions dependency not available - storage backend will not work")
    
    is_valid = len(issues) == 0
    return is_valid, issues

# Initialize storage backend if available
def setup_storage(sandbox_id: str = "ai-session-manager", default_ttl_hours: int = 24):
    """
    Set up the CHUK Sessions storage backend.
    
    Args:
        sandbox_id: Sandbox ID for CHUK Sessions
        default_ttl_hours: Default TTL for sessions in hours
        
    Returns:
        SessionStorage instance or None if setup failed
    """
    if setup_chuk_sessions_storage is not None:
        try:
            return setup_chuk_sessions_storage(
                sandbox_id=sandbox_id,
                default_ttl_hours=default_ttl_hours
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to set up CHUK Sessions storage: {e}")
            return None
    else:
        import warnings
        warnings.warn("CHUK Sessions storage not available - imports failed")
        return None

# Version info
__author__ = "CHUK AI Session Manager Team"
__email__ = "support@chuk.ai"
__license__ = "MIT"
__description__ = "AI Session Management with CHUK Sessions Backend"

# Package metadata
__package_info__ = {
    "name": "chuk_ai_session_manager",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "license": __license__,
    "dependencies": {
        "required": ["pydantic", "uuid", "datetime", "asyncio"],
        "optional": ["tiktoken", "chuk_sessions", "chuk_tool_processor"]
    },
    "import_status": _imported_modules
}