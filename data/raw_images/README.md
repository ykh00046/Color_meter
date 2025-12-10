# Raw Images Directory

This directory contains raw test images for contact lens inspection.

## Structure

```
raw_images/
├── sample_001.jpg        # Example sample images
├── sample_002.jpg
└── ...
```

## Usage

- Place your test images here
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- Recommended resolution: 5MP+ (lens diameter should be >1000 pixels)

## Naming Convention

```
<sku>_<date>_<serial>.jpg
```

Example: `SKU001_20251210_0001.jpg`

## Git Ignore

Note: Most images in this directory are ignored by Git (see `.gitignore`).
Only `sample_*.jpg` files are tracked for testing purposes.
