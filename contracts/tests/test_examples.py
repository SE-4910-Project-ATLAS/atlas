"""Check the message schemas in contracts/schemas and the example messages in contracts/examples.

Run from the repository root:

    python -m pytest contracts -q

Every example file is named <schema name>.<case>.json, for example
i2-event.emergency_vehicle.json, and is validated against schemas/<schema name>.schema.json.
"""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS = Path(__file__).resolve().parent.parent
SCHEMA_DIR = CONTRACTS / "schemas"
EXAMPLE_DIR = CONTRACTS / "examples"

SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.json"))
EXAMPLE_FILES = sorted(EXAMPLE_DIR.glob("*.json"))
MESSAGE_SCHEMAS = [p for p in SCHEMA_FILES if p.name != "common.schema.json"]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


SCHEMAS = {p.name.removesuffix(".schema.json"): load(p) for p in SCHEMA_FILES}
# All schemas are registered under their $id, so that "$ref": "common.schema.json#/..."
# resolves to the local file and nothing is downloaded.
REGISTRY = Registry().with_resources(
    (schema["$id"], Resource.from_contents(schema)) for schema in SCHEMAS.values()
)


def validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(SCHEMAS[name], registry=REGISTRY)


def errors(name: str, message: dict) -> list[str]:
    return [e.message for e in validator(name).iter_errors(message)]


def example(file_name: str) -> dict:
    return copy.deepcopy(load(EXAMPLE_DIR / file_name))


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_json_schema(path: Path) -> None:
    Draft202012Validator.check_schema(load(path))


@pytest.mark.parametrize("path", MESSAGE_SCHEMAS, ids=lambda p: p.name)
def test_every_message_schema_has_an_example(path: Path) -> None:
    name = path.name.removesuffix(".schema.json")
    assert any(p.name.startswith(name + ".") for p in EXAMPLE_FILES), f"No example for {name}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_is_valid(path: Path) -> None:
    name = path.name.split(".")[0]
    assert name in SCHEMAS, f"{path.name}: no schema {name}.schema.json"
    assert errors(name, load(path)) == []


# Messages that must be rejected: each one breaks one rule of a schema.
INVALID = [
    ("i2-event", "i2-event.emergency_vehicle.json", {"vehicle_type": None}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"ts": "2026-11-26T13:15:30+03:00"}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"beacon_confidence": 1.5}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"confidence": None}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"confidence": 1.2}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"direction": 360}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"camera_id": "IZUM 001"}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"event_id": "not-a-uuid"}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"extra_field": 1}),
    ("i2-event", "i2-event.stalled_vehicle.json", {"lanes": []}),
    ("i2-event", "i2-event.stalled_vehicle.json", {"vehicle_type": "ambulance"}),
    ("i2-event", "i2-event.stalled_vehicle.json", {"beacon_confidence": 0.5}),
    ("i3-camera-status", "i3-camera-status.offline.json", {"status": "alert"}),
    ("i7-driver-messages", "i7-driver-messages.alert.json", {"msg": "alarm"}),
    ("i7-driver-messages", "i7-driver-messages.alert.json", {"text": ""}),
    ("i7-driver-messages", "i7-driver-messages.ack.json", {"ts": "2026-11-26T10:15:30.120Z"}),
    ("i7-driver-messages", "i7-driver-messages.route_offer.json", {"time_saved_s": -60}),
    ("i9-dashboard-messages", "i9-dashboard-messages.alert_card.json", {"vehicle_type": None}),
    ("i9-dashboard-messages", "i9-dashboard-messages.zone_update.json", {"drivers_warned": -1}),
    ("i9-dashboard-messages", "i9-dashboard-messages.camera_status.json", {"status": "unknown"}),
]


@pytest.mark.parametrize(
    "schema, file_name, change",
    INVALID,
    ids=[f"{file}:{next(iter(change))}" for _, file, change in INVALID],
)
def test_invalid_message_is_rejected(schema: str, file_name: str, change: dict) -> None:
    message = example(file_name)
    message.update(change)
    assert errors(schema, message) != []


# Messages that must be accepted: allowed variants of the examples.
VALID = [
    ("i2-event", "i2-event.emergency_vehicle.json", {"beacon_confidence": 0.93}),
    ("i2-event", "i2-event.emergency_vehicle.json", {"snapshot_id": None}),
    ("i2-event", "i2-event.stalled_vehicle.json", {"confidence": 0.75}),
    ("i9-dashboard-messages", "i9-dashboard-messages.alert_card.json", {"snapshot_id": None}),
]


@pytest.mark.parametrize(
    "schema, file_name, change",
    VALID,
    ids=[f"{file}:{next(iter(change))}" for _, file, change in VALID],
)
def test_valid_variant_is_accepted(schema: str, file_name: str, change: dict) -> None:
    message = example(file_name)
    message.update(change)
    assert errors(schema, message) == []


def test_missing_required_field_is_rejected() -> None:
    message = example("i2-event.emergency_vehicle.json")
    del message["ts"]
    assert errors("i2-event", message) != []


def test_zone_outside_wgs84_range_is_rejected() -> None:
    message = example("i9-dashboard-messages.zone_update.json")
    message["zone"]["coordinates"][0][0] = [200.0, 38.4]
    assert errors("i9-dashboard-messages", message) != []
