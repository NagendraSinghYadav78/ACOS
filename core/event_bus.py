"""
core/event_bus.py

In-process publish/subscribe bus for inter-agent messaging. Agents
don't call each other directly -- they emit typed Events and other
agents (or the orchestrator) subscribe to the topics they care about.

Supports synchronous dispatch, per-topic subscriber lists, a durable
event log for audit/replay, and basic error isolation so one bad
subscriber can't take down the bus.
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("acos.event_bus")


@dataclass
class Event:
    topic: str
    payload: Dict[str, Any]
    source: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


class EventBus:
    """A synchronous, in-process pub/sub bus with an audit log.

    Design notes:
    - Subscribers are plain Python callables: fn(event: Event) -> None
    - Wildcard subscriptions ("*") receive every event, used by the
      MonitoringLayer / GovernanceAgent for audit trails.
    - The bus keeps a bounded in-memory log; callers needing durable
      storage should also persist relevant events into SharedMemory
      (see core/memory.py), which the Orchestrator does by default.
    """

    def __init__(self, max_log_size: int = 5000):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._log: List[Event] = []
        self._max_log_size = max_log_size

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(topic, []).append(handler)
        logger.debug("Subscribed handler to topic=%s", topic)

    def unsubscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        if topic in self._subscribers and handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)

    def publish(self, topic: str, payload: Dict[str, Any], source: str,
                correlation_id: Optional[str] = None) -> Event:
        event = Event(topic=topic, payload=payload, source=source,
                       correlation_id=correlation_id)
        self._append_log(event)

        handlers = list(self._subscribers.get(topic, []))
        handlers += list(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # error isolation
                logger.exception("Subscriber failed for topic=%s: %s", topic, exc)
        return event

    def _append_log(self, event: Event) -> None:
        self._log.append(event)
        if len(self._log) > self._max_log_size:
            self._log = self._log[-self._max_log_size:]

    def history(self, topic: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        items = self._log if topic is None else [e for e in self._log if e.topic == topic]
        return [e.to_dict() for e in items[-limit:]]
