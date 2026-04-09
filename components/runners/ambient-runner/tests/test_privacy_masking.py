#!/usr/bin/env python3
"""
Test privacy masking function for Langfuse observability.

Validates that:
1. User messages and assistant responses are redacted (all non-empty strings)
2. Usage metrics (tokens, costs) are preserved
3. Metadata is traversed and string values inside it are redacted where not allow-listed
4. Nested structures are handled correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ambient_runner.observability import _privacy_masking_function


def test_string_masking():
    """Test that non-empty strings are redacted at the top level."""
    assert _privacy_masking_function("") == ""
    assert _privacy_masking_function("error") == "[REDACTED FOR PRIVACY]"
    assert _privacy_masking_function("claude-3-5-sonnet") == "[REDACTED FOR PRIVACY]"

    long_text = "This is a user message that contains sensitive information about their business"
    assert _privacy_masking_function(long_text) == "[REDACTED FOR PRIVACY]"


def test_dict_usage_preservation():
    """Test that usage metrics are preserved in dictionaries."""
    usage_data = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 100,
        "total_tokens": 1800,
        "cost_usd": 0.05,
    }

    masked = _privacy_masking_function(usage_data)

    # All usage fields should be preserved
    assert masked["input_tokens"] == 1000
    assert masked["output_tokens"] == 500
    assert masked["cache_read_input_tokens"] == 200
    assert masked["cache_creation_input_tokens"] == 100
    assert masked["total_tokens"] == 1800
    assert masked["cost_usd"] == 0.05


def test_dict_content_masking():
    """Test that content fields are redacted in dictionaries."""
    message_data = {
        "role": "user",
        "content": "Please help me analyze this confidential document with sensitive customer data",
        "model": "claude-3-5-sonnet",
        "turn": 1,
    }

    masked = _privacy_masking_function(message_data)

    # Content should be redacted
    assert masked["content"] == "[REDACTED FOR PRIVACY]"

    assert masked["role"] == "user"
    assert masked["model"] == "claude-3-5-sonnet"
    assert masked["turn"] == 1


def test_nested_structure():
    """Test masking of nested dictionaries and lists."""
    trace_data = {
        "input": [
            {
                "role": "user",
                "content": "Here is my confidential business plan with trade secrets and financial projections",
            }
        ],
        "output": "Based on your business plan, I recommend the following strategies for growth and expansion",
        "usage": {"input_tokens": 500, "output_tokens": 250},
        "metadata": {
            "model": "claude-sonnet-4-5@20250929",
            "turn": 2,
            "session_id": "session-123",
        },
    }

    masked = _privacy_masking_function(trace_data)

    # Input content should be redacted
    assert masked["input"][0]["content"] == "[REDACTED FOR PRIVACY]"
    assert masked["input"][0]["role"] == "user"

    # Output should be redacted
    assert masked["output"] == "[REDACTED FOR PRIVACY]"

    # Usage should be preserved
    assert masked["usage"]["input_tokens"] == 500
    assert masked["usage"]["output_tokens"] == 250

    assert masked["metadata"]["model"] == "claude-sonnet-4-5@20250929"
    assert masked["metadata"]["turn"] == 2
    assert masked["metadata"]["session_id"] == "session-123"


def test_list_masking():
    """Test masking of list items."""
    messages = [
        "Short metadata value",
        "This is a very long user message that contains sensitive personal information about the user",
        {"content": "Another sensitive message in a nested dictionary structure"},
    ]

    masked = _privacy_masking_function(messages)

    assert masked[0] == "[REDACTED FOR PRIVACY]"

    assert masked[1] == "[REDACTED FOR PRIVACY]"

    # Nested dict content redacted
    assert masked[2]["content"] == "[REDACTED FOR PRIVACY]"


def test_tool_tracking_data():
    """Test that tool tracking data is handled correctly."""
    tool_data = {
        "tool_name": "Read",
        "tool_id": "toolu_abc123",
        "input": {"file_path": "/workspace/src/main.py"},
        "output": {
            "result": "File contents here with potentially sensitive code and comments about implementation details"
        },
        "is_error": False,
        "metadata": {"turn": 3},
    }

    masked = _privacy_masking_function(tool_data)

    # Tool metadata preserved
    assert masked["tool_name"] == "Read"
    assert masked["tool_id"] == "toolu_abc123"
    assert masked["is_error"] is False
    assert masked["metadata"]["turn"] == 3

    assert masked["input"]["file_path"] == "[REDACTED FOR PRIVACY]"

    # Tool output result redacted (long content)
    assert masked["output"]["result"] == "[REDACTED FOR PRIVACY]"


def test_primitive_types():
    """Test that primitive types are preserved."""
    assert _privacy_masking_function(42) == 42
    assert _privacy_masking_function(3.14) == 3.14
    assert _privacy_masking_function(True) is True
    assert _privacy_masking_function(False) is False
    assert _privacy_masking_function(None) is None


def test_empty_structures():
    """Test that empty structures are handled correctly."""
    assert _privacy_masking_function({}) == {}
    assert _privacy_masking_function([]) == []
    assert _privacy_masking_function("") == ""


def test_real_world_trace():
    """Test with realistic Langfuse trace structure."""
    trace = {
        "name": "claude_interaction",
        "input": [
            {
                "role": "user",
                "content": "Can you help me refactor this legacy codebase? It has several security vulnerabilities.",
            }
        ],
        "output": "I'll help you refactor the codebase. First, let me analyze the current structure and identify the security issues.",
        "model": "claude-sonnet-4-5@20250929",
        "usage_details": {
            "input": 150,
            "output": 75,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
        },
        "metadata": {
            "turn": 1,
            "session_id": "test-session-123",
            "namespace": "prod-namespace",
        },
    }

    masked = _privacy_masking_function(trace)

    assert masked["name"] == "[REDACTED FOR PRIVACY]"

    # User input redacted
    assert masked["input"][0]["content"] == "[REDACTED FOR PRIVACY]"

    # Assistant output redacted
    assert masked["output"] == "[REDACTED FOR PRIVACY]"

    # Model preserved
    assert masked["model"] == "claude-sonnet-4-5@20250929"

    # Usage details fully preserved (this is what we track!)
    assert masked["usage_details"]["input"] == 150
    assert masked["usage_details"]["output"] == 75
    assert masked["usage_details"]["cache_read_input_tokens"] == 50
    assert masked["usage_details"]["cache_creation_input_tokens"] == 25

    assert masked["metadata"]["turn"] == 1
    assert masked["metadata"]["session_id"] == "test-session-123"
    assert masked["metadata"]["namespace"] == "[REDACTED FOR PRIVACY]"


if __name__ == "__main__":
    print("Testing Langfuse privacy masking function...")
    print("=" * 60)

    tests = [
        ("String masking", test_string_masking),
        ("Usage preservation", test_dict_usage_preservation),
        ("Content masking", test_dict_content_masking),
        ("Nested structure", test_nested_structure),
        ("List masking", test_list_masking),
        ("Tool tracking data", test_tool_tracking_data),
        ("Primitive types", test_primitive_types),
        ("Empty structures", test_empty_structures),
        ("Real-world trace", test_real_world_trace),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: Unexpected error: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All privacy masking tests passed!")
        print(
            "User messages and responses will be redacted while preserving usage metrics."
        )
