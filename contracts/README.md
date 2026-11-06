# Contracts

Shared interfaces used by more than one sub-system, for example:

- the event message that the vision service sends to the backend,
- the REST / WebSocket API used by the mobile app and the dashboard.

Keeping them in one place makes sure all sub-systems use the same version.
A change here should be reviewed by the owners of every sub-system that uses it.
