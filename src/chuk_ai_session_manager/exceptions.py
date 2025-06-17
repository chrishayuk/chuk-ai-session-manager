# chuk_ai_session_manager/exceptions.py
"""
Exception classes for the chuk session manager.

This module defines the exception hierarchy used throughout the 
session manager to provide specific, informative error conditions
for various failure modes.
"""

class SessionManagerError(Exception):
    """
    Base exception for all session manager errors.
    
    All other session manager exceptions inherit from this class,
    making it easy to catch all session-related errors with a single
    except clause if needed.
    """
    pass


class SessionNotFound(SessionManagerError):
    """
    Raised when the requested session ID is not found in storage.
    
    This exception is typically raised when:
    - Attempting to retrieve a session with an invalid ID
    - Accessing a session that has been deleted
    - Using an ID that does not conform to expected format
    """
    def __init__(self, session_id=None, message=None):
        self.session_id = session_id
        default_message = f"Session not found: {session_id}" if session_id else "Session not found"
        super().__init__(message or default_message)


class SessionAlreadyExists(SessionManagerError):
    """
    Raised when attempting to create a session with an ID that already exists.
    
    This exception is typically raised during session creation when:
    - Explicitly setting an ID that conflicts with an existing session
    - A UUID collision occurs (extremely rare)
    """
    def __init__(self, session_id=None, message=None):
        self.session_id = session_id
        default_message = f"Session already exists: {session_id}" if session_id else "Session already exists"
        super().__init__(message or default_message)


class InvalidSessionOperation(SessionManagerError):
    """
    Raised when attempting an invalid operation on a session.
    
    This exception is typically raised when:
    - Performing operations on a closed or archived session
    - Adding events with incorrect sequencing or relationships
    - Attempting unsupported operations in the current session state
    """
    def __init__(self, operation=None, reason=None, message=None):
        self.operation = operation
        self.reason = reason
        
        if message:
            default_message = message
        elif operation and reason:
            default_message = f"Invalid operation '{operation}': {reason}"
        elif operation:
            default_message = f"Invalid operation: {operation}"
        else:
            default_message = "Invalid session operation"
            
        super().__init__(default_message)


class TokenLimitExceeded(SessionManagerError):
    """
    Raised when a token limit is exceeded in a session operation.
    
    This exception is typically raised when:
    - Adding content that would exceed configured token limits
    - Attempting to generate a prompt that exceeds model token limits
    """
    def __init__(self, limit=None, actual=None, message=None):
        self.limit = limit
        self.actual = actual
        
        if message:
            default_message = message
        elif limit and actual:
            default_message = f"Token limit exceeded: {actual} > {limit}"
        else:
            default_message = "Token limit exceeded"
            
        super().__init__(default_message)


class StorageError(SessionManagerError):
    """
    Raised when a session storage operation fails.
    
    This is a base class for more specific storage errors.
    It can be raised directly for general storage failures.
    """
    pass


class ToolProcessingError(SessionManagerError):
    """
    Raised when tool processing fails in a session.
    
    This exception is typically raised when:
    - A tool execution fails after all retries
    - Invalid tool parameters are provided
    - Tool results cannot be properly processed
    """
    def __init__(self, tool_name=None, reason=None, message=None):
        self.tool_name = tool_name
        self.reason = reason
        
        if message:
            default_message = message
        elif tool_name and reason:
            default_message = f"Tool '{tool_name}' processing error: {reason}"
        elif tool_name:
            default_message = f"Tool '{tool_name}' processing error"
        else:
            default_message = "Tool processing error"
            
        super().__init__(default_message)