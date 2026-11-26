# Contracts

Shared message formats used by more than one sub-system. Every format is a
[JSON Schema](https://json-schema.org/) (Draft 2020-12) in `schemas/`, with example
messages in `examples/` and a test that checks both (`tests/`). The interfaces are
described on the Confluence page **System Architecture**, Section 6.

A change here should be reviewed by the owners of every sub-system that uses it.

## Schemas

| Schema | Interface | From → To | Channel | Messages (`msg`) |
| --- | --- | --- | --- | --- |
| `i2-event.schema.json` | I2 | Vision service → Backend | Redis stream `atlas:events`; `POST /events` in deployment Option B | Emergency-vehicle, stalled-vehicle and accident event (`type`) |
| `i3-camera-status.schema.json` | I3 | Vision service → Backend | Redis stream `atlas:camera-status` | Camera online / offline |
| `i7-driver-messages.schema.json` | I7 | Backend ↔ Driver app | WebSocket (Driving Mode); the same `alert` through FCM (I8) | `alert`, `ack`, `all_clear`, `route_offer` (draft) |
| `i9-dashboard-messages.schema.json` | I9 | Backend → Dashboard | WebSocket `/ws/operator` | `alert_card`, `camera_status`, `zone_update` |
| `common.schema.json` | – | – | – | Shared definitions (ids, UTC time, camera id, lanes); not a message |

Not written yet: I3b density (with the HLF-3 work), I6 driver location, I11 route-offer
outcome and I13 routes (with the HLF-6 work), and the REST API of the dashboard.

## Rules for all messages

- Every message has `schema_version` (an integer, now `1`).
- I7 and I9 messages have a `msg` field that tells which message it is.
- Times (`ts`) are UTC in ISO 8601 with `Z`, for example `2026-11-26T10:15:30.120Z`.
  In an I2 event, `ts` is the time of the camera frame that confirmed the event, not
  the time the message was sent (technical meeting of 21.11.2026).
- Ids (`event_id`, `snapshot_id`, `offer_id`) are lower-case UUIDs. The vision service
  keeps the same `event_id` for the same tracked vehicle, so repeated messages do not
  raise new alerts.
- Units: `speed` in km/h, `direction` in degrees clockwise from north, zones as GeoJSON
  polygons in WGS 84 (`[longitude, latitude]`).
- Every field listed in a schema is present; a field that does not apply is `null`.
  Fields that are not in the schema are rejected (`additionalProperties: false`).
- No personal data: messages to the dashboard carry counts of drivers, never their
  positions (NFR-21).

**On the wire.** A message is sent as one JSON object:

- Redis Streams: one entry per message with a single field `data` that holds the JSON
  text, for example `XADD atlas:events MAXLEN ~ 10000 * data '{"schema_version":1,...}'`.
- WebSocket: one text frame per message.
- FCM: the JSON of the `alert` in the data payload, with the same `event_id`.

## Examples and tests

Each file in `examples/` is named `<schema name>.<case>.json` (for example
`i2-event.accident.json`) and must be valid against `schemas/<schema name>.schema.json`.
The test checks that every schema is a valid JSON Schema, that every message schema
has an example, that every example is valid, and that a set of wrong messages is
rejected.

Run the tests from the repository root, in a virtual environment outside the
repository:

```bash
python -m venv ~/.venvs/atlas-contracts
~/.venvs/atlas-contracts/bin/python -m pip install -r contracts/requirements.txt
~/.venvs/atlas-contracts/bin/python -m pytest contracts -q
```

On Windows the Python of the environment is `$HOME\.venvs\atlas-contracts\Scripts\python.exe`.

`requirements.in` lists the direct dependencies; `requirements.txt` is the lock file,
generated inside the pinned Python image as described in "Updating dependencies" of the
root README:

```bash
docker run --rm -v "$PWD/contracts:/w" -w /w python:3.15.2-slim \
  sh -c "pip install -q -r requirements.in && pip freeze > requirements.txt"
```

## Changing a schema

1. Change the schema in a Pull Request linked to a Jira issue, and update the examples
   and the test in the same Pull Request.
2. Increase `schema_version` (the `const` in the schema) when a field is added, removed,
   renamed or its meaning or allowed values change. Description-only changes keep the
   version.
3. Add as reviewers the owners of every sub-system that sends or receives the message
   (see the table above and `.github/CODEOWNERS`); they update their code to the new
   version.
4. Update Section 6 of the System Architecture page if the interface changes.
