# UI Flow Diagram (Text)

This is a compact text diagram for the UI flow.

```
STD Register
  -> STD_REGISTRATION Validate
       -> (STD_ACCEPTABLE) -> Activate
       -> (STD_RETAKE)     -> Re-capture
       -> (STD_UNSTABLE)   -> Re-define groups

Activate
  -> sets ACTIVE model pointers (SKU+INK)

Inspection
  -> requires ACTIVE
  -> OK / NG_COLOR / NG_PATTERN / RETAKE
```

Key rules:
- Activation is separate from registration.
- INSPECTION without ACTIVE returns RETAKE.
- LOW/MID/HIGH are user-defined groups (no strict order).
