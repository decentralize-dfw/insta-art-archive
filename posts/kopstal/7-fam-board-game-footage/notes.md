# MERGED-20sec.mp4

Perspective-corrected derivative. Original source unchanged on disk.

Canvas image is mapped into the same fixed rectangle in every frame: [500, 130, 1300, 1040] (x, y, width, height). Corner mapping error is numerical transform precision, not a claim of zero physical feature-tracking error.

Source duration, 823 video frames, 2300 × 1300 output and 30 fps are preserved. Audio stream is copied; decoded audio SHA-256 matches the source. No generative fill, colour grading, speed changes, missing-frame replacement or still-image substitution.

The final geometry uses direct robust outside-edge fits for each frame, with no accumulated feature pose. A three-frame median filters corner measurements only. Preliminary surface-feature tracking was rejected because the moving hand could be mistaken for camera movement.

Black corner gaps, pre-existing clipping, blur and exposure changes baked into the input remain present. They are not repaired with invented image data. The outer output rectangle is fixed; existing image defects inside it are not presented as restored painting.

Source SHA-256: `4f49f64abe3b9d4b9de0ae82cfb3b65a29dd9767131c3ae253ac4fb2639eb3a4`

Archive SHA-256: `1ef26ec01c7ac815268bd37051fa67b8c0dbd5a3f9e30292d78d711d506da6fa`

Validation and per-frame geometry: [processing report](../../../processing/canvas-lock-2026-11-10/REPORT.md).
