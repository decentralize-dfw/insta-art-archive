# MERGED-20sec.mp4 · content lock V2

Replaces the rejected perimeter-only V1. Painting landmarks, physically framed stage references and completed-painting corners define one projective transform for every source frame. The four input corners for all 823 frames are recorded in the processing report.

Each output frame contains only a geometric resampling of the same original frame. No AI, generative fill, inpainting, clone fill, temporal pixel replacement or still-image substitution. Missing/clipped areas remain empty or retain the defects already baked into the source.

Native dimensions, 30 fps, all frames, duration and original decoded audio are preserved. H.264 CRF17, 25 Mbit/s ceiling, FastStart.

Independent painting-point test: median 1.057 output pixels; 95th percentile of pair medians 2.263 pixels. This is an observed residual, not a zero-motion guarantee. Low-texture/occluded frames use neighboring geometric reference poses; no missing image data is invented.

SHA-256: `135fa470a98dd1d11be9023c7c9a519c89fdb65fa793cf52ed10dbcc1bb6ca71`

[Per-frame corners, checks and method](../../../processing/canvas-content-lock-v2/REPORT.md).
