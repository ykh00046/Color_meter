# Logs Directory

This directory contains application log files.

## Structure

```
logs/
├── app_20251210.log          # Daily application logs
├── inspection_20251210.log   # Inspection event logs
└── error_20251210.log        # Error logs
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (non-critical issues)
- **ERROR**: Error messages (critical issues)
- **CRITICAL**: Critical errors (system failures)

## Rotation

Logs are automatically rotated:
- Daily rotation at midnight
- Maximum 30 days retention
- Compressed after 7 days

## Git Ignore

All `.log` files are ignored by Git (see `.gitignore`).

## Usage

View recent logs:
```bash
tail -f logs/app_$(date +%Y%m%d).log
```

Search for errors:
```bash
grep -i "error" logs/app_*.log
```
