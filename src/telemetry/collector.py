"""
Telemetry collector supporting 9 sources from the paper:
network_flow, process_exec, cloud_audit, container, authentication,
application, dns, vpn, file_access.

In production these attach to eBPF/ETW/CloudTrail/Kafka.
This module provides both a simulated generator (for testing) and a
Kafka consumer (for production).
"""
from __future__ import annotations
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Callable

from ..common.models import TelemetryEvent
from ..common.config import TelemetrySource, AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Lateral-movement-capable processes per the paper
LATERAL_MOVEMENT_TOOLS = ["psexec", "ssh", "rdp", "wmi", "winrm", "dcom", "powershell"]
NORMAL_PROCESSES = ["chrome", "outlook", "excel", "python", "java", "nginx", "postgres"]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
CLOUD_PROVIDERS = ["aws", "azure", "onprem"]


class SimulatedTelemetryGenerator:
    """
    Generates synthetic telemetry events for a simulated 10,000-node enterprise
    matching the paper's testbed: 6,500 servers, 2,500 containers, 1,000 laptops,
    6,000 users, 40 critical databases.
    """

    def __init__(
        self,
        num_workloads: int = 10_000,
        num_users: int = 6_000,
        num_databases: int = 40,
        event_rate_per_sec: float = 5_000.0,
        attack_probability: float = 0.001,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        self.num_workloads = num_workloads
        self.num_users = num_users
        self.num_databases = num_databases
        self.event_rate_per_sec = event_rate_per_sec
        self.attack_probability = attack_probability

        self._workload_ids = [f"wl-{i:05d}" for i in range(num_workloads)]
        self._user_ids = [f"usr-{i:04d}" for i in range(num_users)]
        self._db_ids = [f"db-{i:02d}" for i in range(num_databases)]
        self._all_nodes = self._workload_ids + self._user_ids + self._db_ids
        self._critical_assets = set(self._db_ids)

    def _random_node(self) -> str:
        return random.choice(self._all_nodes)

    def _is_business_hours(self, ts: datetime) -> bool:
        return 8 <= ts.hour <= 18 and ts.weekday() < 5

    def _make_normal_event(self, ts: datetime) -> TelemetryEvent:
        src = random.choice(self._workload_ids + self._user_ids)
        dst = random.choice(self._all_nodes)
        source_type = random.choice(list(TelemetrySource))
        process = random.choice(NORMAL_PROCESSES)
        return TelemetryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=ts,
            source_node=src,
            destination_node=dst,
            event_type=source_type.value,
            protocol=random.choice(PROTOCOLS),
            port=random.choice([22, 80, 443, 3306, 5432, 8080]),
            bytes_transferred=random.randint(100, 1_000_000),
            user=random.choice(self._user_ids),
            process=process,
            success=random.random() > 0.02,
            raw_data={"simulated": True, "business_hours": self._is_business_hours(ts)},
        )

    def _make_attack_event(self, ts: datetime, compromised: str) -> TelemetryEvent:
        """Lateral movement event: compromised node reaches out with admin tools."""
        dst = random.choice(self._workload_ids + self._db_ids)
        process = random.choice(LATERAL_MOVEMENT_TOOLS)
        return TelemetryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=ts,
            source_node=compromised,
            destination_node=dst,
            event_type=TelemetrySource.PROCESS_EXEC.value,
            protocol="TCP",
            port=random.choice([22, 3389, 445, 135]),
            bytes_transferred=random.randint(50, 50_000),
            user=random.choice(self._user_ids),
            process=process,
            success=True,
            raw_data={"simulated": True, "attack": True, "tool": process},
        )

    def generate(
        self,
        duration_sec: float = 60.0,
        start_time: Optional[datetime] = None,
    ) -> Generator[TelemetryEvent, None, None]:
        """Yield synthetic events for `duration_sec` seconds of simulated time."""
        ts = start_time or datetime.utcnow()
        end_ts = ts + timedelta(seconds=duration_sec)
        compromised_nodes: List[str] = []

        while ts < end_ts:
            # Possibly start a new attack chain
            if random.random() < self.attack_probability and not compromised_nodes:
                compromised_nodes.append(random.choice(self._workload_ids))
                logger.debug("Attack started from %s at %s", compromised_nodes[0], ts)

            if compromised_nodes and random.random() < 0.1:
                yield self._make_attack_event(ts, random.choice(compromised_nodes))
            else:
                yield self._make_normal_event(ts)

            # Advance simulated time proportional to target event rate
            ts += timedelta(seconds=1.0 / self.event_rate_per_sec)


class KafkaTelemetryConsumer:
    """Reads normalized telemetry events from a Kafka topic."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        self._consumer = None

    def start(self) -> None:
        try:
            from kafka import KafkaConsumer
            import json

            self._consumer = KafkaConsumer(
                self._cfg.kafka.telemetry_topic,
                bootstrap_servers=self._cfg.kafka.bootstrap_servers,
                group_id=self._cfg.kafka.group_id,
                auto_offset_reset=self._cfg.kafka.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            logger.info("Kafka consumer started on %s", self._cfg.kafka.bootstrap_servers)
        except ImportError:
            logger.warning("kafka-python not installed; KafkaTelemetryConsumer unavailable")

    def consume(self, handler: Callable[[TelemetryEvent], None]) -> None:
        if self._consumer is None:
            raise RuntimeError("Call start() before consume()")
        for msg in self._consumer:
            try:
                data = msg.value
                event = TelemetryEvent(**data)
                handler(event)
            except Exception as exc:
                logger.error("Failed to parse telemetry message: %s", exc)

    def stop(self) -> None:
        if self._consumer:
            self._consumer.close()
