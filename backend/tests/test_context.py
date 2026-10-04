from app.context import compact_conversation
from app.models import MessageRecord


def message(index: int, role: str = "user") -> MessageRecord:
    return MessageRecord(
        id=f"msg_{index}",
        session_id="ses_1",
        role=role,
        content=f"Message {index} " + "detail " * 100,
        created_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def test_compaction_keeps_recent_messages_out_of_summary() -> None:
    messages = [message(index, "assistant" if index % 2 else "user") for index in range(10)]
    summary = compact_conversation(messages, recent_limit=4, max_chars=5000)

    assert "Message 0" in summary
    assert "Message 5" in summary
    assert "Message 6" not in summary
    assert len(summary) <= 5000


def test_compaction_is_empty_when_all_messages_are_recent() -> None:
    assert compact_conversation([message(1)], recent_limit=4, max_chars=1000) == ""
