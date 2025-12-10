# Configuration Directory

This directory contains system and SKU configuration files.

## Structure

```
config/
├── system_config.json           # System-wide configuration
├── system_config.example.json   # Example template
└── sku_db/                      # SKU baseline database
    ├── SKU001.json
    ├── SKU002.json
    └── ...
```

## System Configuration

`system_config.json` contains:
- Camera settings (resolution, exposure, gain)
- Lighting parameters
- Processing thresholds
- MES integration settings

**Important**: This file may contain sensitive information (IP addresses, credentials).
It is ignored by Git. Use `system_config.example.json` as a template.

## SKU Configuration

Each SKU has a JSON file in `sku_db/` containing:
- Zone definitions (LAB values)
- Delta-E thresholds
- Metadata (registration date, sample count)

These files are generated automatically using the SKU registration tool.

## Version Control

- ✅ `system_config.example.json` - Tracked in Git
- ✅ `sku_db/SKU_EXAMPLE.json` - Tracked in Git
- ❌ `system_config.json` - Ignored (contains secrets)
- ⚠️ `sku_db/*.json` - Consider backing up separately
