# tests/test_exceptions.py
"""
Test suite for exception handling in chuk_ai_session_manager.

Tests custom exceptions and error conditions.
"""

import pytest

from chuk_ai_session_manager.exceptions import (
    SessionManagerError,
    SessionNotFound,
    SessionAlreadyExists,
    InvalidSessionOperation,
    TokenLimitExceeded,
    StorageError,
    ToolProcessingError
)


class TestSessionManagerExceptions:
    """Test custom exception classes."""
    
    def test_session_manager_error_base(self):
        """Test base SessionManagerError."""
        error = SessionManagerError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)
    
    def test_session_not_found_basic(self):
        """Test SessionNotFound exception."""
        error = SessionNotFound()
        assert "Session not found" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_session_not_found_with_id(self):
        """Test SessionNotFound with session ID."""
        error = SessionNotFound(session_id="test-session-123")
        assert "test-session-123" in str(error)
        assert error.session_id == "test-session-123"
    
    def test_session_not_found_with_custom_message(self):
        """Test SessionNotFound with custom message."""
        error = SessionNotFound(message="Custom not found message")
        assert str(error) == "Custom not found message"
    
    def test_session_not_found_with_id_and_message(self):
        """Test SessionNotFound with both ID and custom message."""
        error = SessionNotFound(session_id="test-123", message="Custom message")
        assert str(error) == "Custom message"
        assert error.session_id == "test-123"
    
    def test_session_already_exists_basic(self):
        """Test SessionAlreadyExists exception."""
        error = SessionAlreadyExists()
        assert "Session already exists" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_session_already_exists_with_id(self):
        """Test SessionAlreadyExists with session ID."""
        error = SessionAlreadyExists(session_id="duplicate-session")
        assert "duplicate-session" in str(error)
        assert error.session_id == "duplicate-session"
    
    def test_session_already_exists_with_custom_message(self):
        """Test SessionAlreadyExists with custom message."""
        error = SessionAlreadyExists(message="Custom exists message")
        assert str(error) == "Custom exists message"
    
    def test_invalid_session_operation_basic(self):
        """Test InvalidSessionOperation exception."""
        error = InvalidSessionOperation()
        assert "Invalid session operation" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_invalid_session_operation_with_operation(self):
        """Test InvalidSessionOperation with operation name."""
        error = InvalidSessionOperation(operation="add_event")
        assert "add_event" in str(error)
        assert error.operation == "add_event"
    
    def test_invalid_session_operation_with_reason(self):
        """Test InvalidSessionOperation with operation and reason."""
        error = InvalidSessionOperation(
            operation="close_session", 
            reason="Session is already closed"
        )
        assert "close_session" in str(error)
        assert "Session is already closed" in str(error)
        assert error.operation == "close_session"
        assert error.reason == "Session is already closed"
    
    def test_invalid_session_operation_with_custom_message(self):
        """Test InvalidSessionOperation with custom message."""
        error = InvalidSessionOperation(message="Custom operation error")
        assert str(error) == "Custom operation error"
    
    def test_token_limit_exceeded_basic(self):
        """Test TokenLimitExceeded exception."""
        error = TokenLimitExceeded()
        assert "Token limit exceeded" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_token_limit_exceeded_with_limits(self):
        """Test TokenLimitExceeded with limit values."""
        error = TokenLimitExceeded(limit=1000, actual=1500)
        assert "1000" in str(error)
        assert "1500" in str(error)
        assert error.limit == 1000
        assert error.actual == 1500
    
    def test_token_limit_exceeded_with_custom_message(self):
        """Test TokenLimitExceeded with custom message."""
        error = TokenLimitExceeded(message="Custom token limit error")
        assert str(error) == "Custom token limit error"
    
    def test_storage_error_basic(self):
        """Test StorageError exception."""
        error = StorageError("Storage connection failed")
        assert "Storage connection failed" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_tool_processing_error_basic(self):
        """Test ToolProcessingError exception."""
        error = ToolProcessingError()
        assert "Tool processing error" in str(error)
        assert isinstance(error, SessionManagerError)
    
    def test_tool_processing_error_with_tool_name(self):
        """Test ToolProcessingError with tool name."""
        error = ToolProcessingError(tool_name="calculator")
        assert "calculator" in str(error)
        assert error.tool_name == "calculator"
    
    def test_tool_processing_error_with_reason(self):
        """Test ToolProcessingError with tool name and reason."""
        error = ToolProcessingError(
            tool_name="search", 
            reason="API key invalid"
        )
        assert "search" in str(error)
        assert "API key invalid" in str(error)
        assert error.tool_name == "search"
        assert error.reason == "API key invalid"
    
    def test_tool_processing_error_with_custom_message(self):
        """Test ToolProcessingError with custom message."""
        error = ToolProcessingError(message="Custom tool error")
        assert str(error) == "Custom tool error"


class TestExceptionHierarchy:
    """Test exception inheritance and hierarchy."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from SessionManagerError."""
        exceptions = [
            SessionNotFound(),
            SessionAlreadyExists(),
            InvalidSessionOperation(),
            TokenLimitExceeded(),
            StorageError("test"),
            ToolProcessingError()
        ]
        
        for exc in exceptions:
            assert isinstance(exc, SessionManagerError)
            assert isinstance(exc, Exception)
    
    def test_catch_all_session_errors(self):
        """Test catching all session errors with base exception."""
        errors_to_test = [
            SessionNotFound("test"),
            SessionAlreadyExists("test"),
            InvalidSessionOperation("test"),
            TokenLimitExceeded(message="test"),
            StorageError("test"),
            ToolProcessingError(message="test")
        ]
        
        caught_errors = []
        
        for error in errors_to_test:
            try:
                raise error
            except SessionManagerError as e:
                caught_errors.append(type(e))
            except Exception:
                pytest.fail(f"Should have caught {type(error)} as SessionManagerError")
        
        assert len(caught_errors) == len(errors_to_test)
    
    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        # Test SessionNotFound
        try:
            raise SessionNotFound("session-123")
        except SessionNotFound as e:
            assert e.session_id == "session-123"
        
        # Test TokenLimitExceeded
        try:
            raise TokenLimitExceeded(limit=100, actual=150)
        except TokenLimitExceeded as e:
            assert e.limit == 100
            assert e.actual == 150
        
        # Test ToolProcessingError
        try:
            raise ToolProcessingError(tool_name="test_tool", reason="failed")
        except ToolProcessingError as e:
            assert e.tool_name == "test_tool"
            assert e.reason == "failed"


class TestExceptionUsageScenarios:
    """Test realistic exception usage scenarios."""
    
    def test_session_not_found_in_storage(self):
        """Test SessionNotFound in storage operations."""
        def get_session(session_id):
            if session_id == "nonexistent":
                raise SessionNotFound(session_id=session_id)
            return {"id": session_id}
        
        # Should work for existing session
        result = get_session("existing-session")
        assert result["id"] == "existing-session"
        
        # Should raise for nonexistent session
        with pytest.raises(SessionNotFound) as exc_info:
            get_session("nonexistent")
        
        assert exc_info.value.session_id == "nonexistent"
        assert "nonexistent" in str(exc_info.value)
    
    def test_token_limit_in_prompt_building(self):
        """Test TokenLimitExceeded in prompt building."""
        def build_prompt(tokens_used, token_limit):
            if tokens_used > token_limit:
                raise TokenLimitExceeded(
                    limit=token_limit,
                    actual=tokens_used,
                    message=f"Prompt requires {tokens_used} tokens but limit is {token_limit}"
                )
            return "valid_prompt"
        
        # Should work within limit
        prompt = build_prompt(800, 1000)
        assert prompt == "valid_prompt"
        
        # Should raise when exceeding limit
        with pytest.raises(TokenLimitExceeded) as exc_info:
            build_prompt(1200, 1000)
        
        assert exc_info.value.limit == 1000
        assert exc_info.value.actual == 1200
        assert "1200" in str(exc_info.value)
        assert "1000" in str(exc_info.value)
    
    def test_invalid_operation_on_session(self):
        """Test InvalidSessionOperation for session state violations."""
        class MockSession:
            def __init__(self):
                self.closed = False
            
            def add_event(self, event):
                if self.closed:
                    raise InvalidSessionOperation(
                        operation="add_event",
                        reason="Cannot add events to a closed session"
                    )
                return "event_added"
            
            def close(self):
                self.closed = True
        
        session = MockSession()
        
        # Should work on open session
        result = session.add_event("test_event")
        assert result == "event_added"
        
        # Close session
        session.close()
        
        # Should raise on closed session
        with pytest.raises(InvalidSessionOperation) as exc_info:
            session.add_event("another_event")
        
        assert exc_info.value.operation == "add_event"
        assert "closed session" in exc_info.value.reason
    
    def test_tool_processing_error_with_retries(self):
        """Test ToolProcessingError in tool execution."""
        class MockToolProcessor:
            def __init__(self):
                self.attempt_count = 0
                self.max_retries = 2
            
            def execute_tool(self, tool_name, args):
                self.attempt_count += 1
                
                if self.attempt_count <= self.max_retries:
                    # Simulate transient error
                    raise ToolProcessingError(
                        tool_name=tool_name,
                        reason=f"Network timeout (attempt {self.attempt_count})"
                    )
                elif tool_name == "broken_tool":
                    # Simulate permanent error
                    raise ToolProcessingError(
                        tool_name=tool_name,
                        reason="Tool configuration invalid"
                    )
                else:
                    return f"success_result_for_{tool_name}"
        
        processor = MockToolProcessor()
        
        # Should eventually succeed after retries
        with pytest.raises(ToolProcessingError) as exc_info:
            processor.execute_tool("working_tool", {})
        assert "attempt 1" in exc_info.value.reason
        
        with pytest.raises(ToolProcessingError) as exc_info:
            processor.execute_tool("working_tool", {})
        assert "attempt 2" in exc_info.value.reason
        
        # Third attempt should succeed
        result = processor.execute_tool("working_tool", {})
        assert result == "success_result_for_working_tool"
        
        # Test permanent failure
        processor.attempt_count = 3  # Skip retries
        with pytest.raises(ToolProcessingError) as exc_info:
            processor.execute_tool("broken_tool", {})
        assert exc_info.value.tool_name == "broken_tool"
        assert "configuration invalid" in exc_info.value.reason
    
    def test_storage_error_handling(self):
        """Test StorageError in storage operations."""
        class MockStorage:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
            
            def save(self, data):
                if self.should_fail:
                    raise StorageError("Database connection lost")
                return "saved"
            
            def load(self, key):
                if self.should_fail:
                    raise StorageError("Failed to read from storage")
                return f"data_for_{key}"
        
        # Test successful operations
        storage = MockStorage(should_fail=False)
        assert storage.save("test_data") == "saved"
        assert storage.load("test_key") == "data_for_test_key"
        
        # Test failed operations
        failed_storage = MockStorage(should_fail=True)
        
        with pytest.raises(StorageError) as exc_info:
            failed_storage.save("test_data")
        assert "Database connection lost" in str(exc_info.value)
        
        with pytest.raises(StorageError) as exc_info:
            failed_storage.load("test_key")
        assert "Failed to read from storage" in str(exc_info.value)


class TestExceptionMessageFormatting:
    """Test exception message formatting and details."""
    
    def test_exception_string_representations(self):
        """Test string representations of exceptions."""
        test_cases = [
            (SessionNotFound(), "Session not found"),
            (SessionNotFound(session_id="test"), "Session not found: test"),
            (SessionAlreadyExists(session_id="dup"), "Session already exists: dup"),
            (InvalidSessionOperation(operation="test"), "Invalid operation: test"),
            (InvalidSessionOperation(operation="test", reason="because"), "Invalid operation 'test': because"),
            (TokenLimitExceeded(limit=100, actual=150), "Token limit exceeded: 150 > 100"),
            (ToolProcessingError(tool_name="calc"), "Tool 'calc' processing error"),
            (ToolProcessingError(tool_name="calc", reason="failed"), "Tool 'calc' processing error: failed"),
            (StorageError("connection failed"), "connection failed"),
        ]
        
        for exception, expected_text in test_cases:
            assert expected_text in str(exception)
    
    def test_exception_attribute_access(self):
        """Test accessing exception attributes."""
        # SessionNotFound
        snf = SessionNotFound(session_id="test-session")
        assert snf.session_id == "test-session"
        
        # SessionAlreadyExists
        sae = SessionAlreadyExists(session_id="existing-session")
        assert sae.session_id == "existing-session"
        
        # InvalidSessionOperation
        iso = InvalidSessionOperation(operation="close", reason="already closed")
        assert iso.operation == "close"
        assert iso.reason == "already closed"
        
        # TokenLimitExceeded
        tle = TokenLimitExceeded(limit=1000, actual=1500)
        assert tle.limit == 1000
        assert tle.actual == 1500
        
        # ToolProcessingError
        tpe = ToolProcessingError(tool_name="search", reason="timeout")
        assert tpe.tool_name == "search"
        assert tpe.reason == "timeout"
    
    def test_exception_none_attributes(self):
        """Test exceptions with None attributes."""
        # Test that None attributes don't break string formatting
        snf = SessionNotFound(session_id=None)
        assert "None" not in str(snf)  # Should use default message
        
        iso = InvalidSessionOperation(operation=None, reason=None)
        assert str(iso) == "Invalid session operation"
        
        tle = TokenLimitExceeded(limit=None, actual=None)
        assert str(tle) == "Token limit exceeded"
        
        tpe = ToolProcessingError(tool_name=None, reason=None)
        assert str(tpe) == "Tool processing error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])