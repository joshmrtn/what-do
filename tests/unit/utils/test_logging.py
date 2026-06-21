import io
import json


def test_log_entry_has_required_fields():
    from src.utils.logging import get_logger

    stream = io.StringIO()
    log = get_logger("test.fields", stream=stream)
    log.info("hello world", component="test_comp", duration_ms=42)

    stream.seek(0)
    entry = json.loads(stream.readline())

    assert "timestamp" in entry
    assert entry["component"] == "test_comp"
    assert entry["severity"] == "INFO"
    assert entry["duration_ms"] == 42
    assert entry["message"] == "hello world"


def test_debug_absent_at_info_level():
    from src.utils.logging import get_logger

    stream = io.StringIO()
    log = get_logger("test.level", stream=stream, level="INFO")
    log.debug("should not appear", component="test", duration_ms=0)

    stream.seek(0)
    assert stream.read() == ""


def test_debug_present_at_debug_level():
    from src.utils.logging import get_logger

    stream = io.StringIO()
    log = get_logger("test.debug", stream=stream, level="DEBUG")
    log.debug("should appear", component="test", duration_ms=0)

    stream.seek(0)
    entry = json.loads(stream.readline())
    assert entry["severity"] == "DEBUG"
    assert entry["message"] == "should appear"


def test_warning_and_error_levels():
    from src.utils.logging import get_logger

    stream = io.StringIO()
    log = get_logger("test.levels", stream=stream)
    log.warning("warn msg", component="c", duration_ms=1)
    log.error("err msg", component="c", duration_ms=2)

    stream.seek(0)
    lines = [json.loads(l) for l in stream.readlines()]
    assert lines[0]["severity"] == "WARNING"
    assert lines[1]["severity"] == "ERROR"
