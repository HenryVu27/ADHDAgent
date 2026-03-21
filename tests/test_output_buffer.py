import pytest
from app.agent.output_buffer import OutputBuffer


def test_buffer_accumulates_tokens():
    buf = OutputBuffer(buffer_size=100)
    buf.add("Hello ")
    buf.add("world")
    assert buf.buffered_text == "Hello world"
    assert not buf.is_full


def test_buffer_signals_full():
    buf = OutputBuffer(buffer_size=10)
    buf.add("Hello world!")  # 12 chars > 10
    assert buf.is_full


def test_buffer_flush_returns_accumulated():
    buf = OutputBuffer(buffer_size=100)
    buf.add("Hello ")
    buf.add("world")
    tokens = buf.flush()
    assert tokens == "Hello world"
    assert buf.flushed


def test_passthrough_after_flush():
    buf = OutputBuffer(buffer_size=10)
    buf.add("Hello world!")  # triggers full
    buf.flush()
    assert buf.should_passthrough("more text")
