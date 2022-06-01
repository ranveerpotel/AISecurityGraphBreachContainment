"""
Normalizes raw telemetry events from all 9 sources into a canonical
TelemetryEvent and extracts node identities for graph ingestion.
"""
from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Optional, Tuple

from ..common.models import TelemetryEvent, NodeAttributes
from ..common.config import NodeType, SensitivityLevel, TelemetrySource

logger = logging.getLogger(__name__)

# Regexes for IP/hostname extraction
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_DB_PORTS = {3306, 5432, 1433, 1521, 27017, 6379}
_SENSITIVE_PROCESSES = {"psexec", "wmic", "powershell", "cmd", "bash", "sh"}


def _infer_node_type(node_id: str, port: int, process: str) -> NodeType:
    if node_id.startswith("db-") or port in _DB_PORTS:
        return NodeType.ASSET
    if node_id.startswith("usr-"):
        return NodeType.USER
    return NodeType.WORKLOAD


def _infer_sensitivity(node_id: str, port: int) -> SensitivityLevel:
    if node_id.startswith("db-") or port in _DB_PORTS:
        return SensitivityLevel.CRITICAL
    if port in {22, 3389, 445}:  # admin ports
        return SensitivityLevel.HIGH
    return SensitivityLevel.LOW


class TelemetryNormalizer:
    """
    Converts raw telemetry messages from diverse sources into canonical
    TelemetryEvent objects and extracts NodeAttribute hints for graph nodes.
    """

    def normalize(self, raw: dict) -> Optional[TelemetryEvent]:
        try:
            return TelemetryEvent(
                event_id=raw.get("event_id", ""),
                timestamp=self._parse_ts(raw.get("timestamp")),
                source_node=self._resolve_node(raw, "source"),
                destination_node=self._resolve_node(raw, "destination"),
                source_ip=raw.get("source_ip", raw.get("src_ip", "")),
                destination_ip=raw.get("destination_ip", raw.get("dst_ip", "")),
                event_type=raw.get("event_type", TelemetrySource.NETWORK_FLOW.value),
                protocol=raw.get("protocol", "TCP").upper(),
                port=int(raw.get("port", raw.get("dst_port", 0))),
                bytes_transferred=int(raw.get("bytes_transferred", raw.get("bytes", 0))),
                user=raw.get("user", raw.get("username", "")),
                process=raw.get("process", raw.get("process_name", "")).lower(),
                success=bool(raw.get("success", raw.get("result", True))),
                raw_data=raw,
            )
        except Exception as exc:
            logger.warning("Normalization failed for event: %s — %s", raw.get("event_id"), exc)
            return None

    def extract_node_attributes(self, event: TelemetryEvent) -> Tuple[NodeAttributes, NodeAttributes]:
        """Return (source_attrs, destination_attrs) derived from the event."""
        src_type = _infer_node_type(event.source_node, 0, event.process)
        dst_type = _infer_node_type(event.destination_node, event.port, event.process)
        src = NodeAttributes(
            node_id=event.source_node,
            node_type=src_type,
            sensitivity=_infer_sensitivity(event.source_node, 0),
            ip_address=event.source_ip,
            is_critical=(src_type == NodeType.ASSET),
        )
        dst = NodeAttributes(
            node_id=event.destination_node,
            node_type=dst_type,
            sensitivity=_infer_sensitivity(event.destination_node, event.port),
            ip_address=event.destination_ip,
            is_critical=(dst_type == NodeType.ASSET),
        )
        return src, dst

    def is_suspicious_process(self, event: TelemetryEvent) -> bool:
        return event.process in _SENSITIVE_PROCESSES

    @staticmethod
    def _resolve_node(raw: dict, direction: str) -> str:
        key_map = {
            "source": ["source_node", "src_node", "source_id", "src_host", "src_ip"],
            "destination": ["destination_node", "dst_node", "dest_id", "dst_host", "dst_ip"],
        }
        for key in key_map[direction]:
            val = raw.get(key)
            if val:
                return str(val)
        return f"unknown-{direction}"

    @staticmethod
    def _parse_ts(value) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(value)
        if isinstance(value, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return datetime.utcnow()
