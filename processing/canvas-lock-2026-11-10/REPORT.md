# Video upload and canvas lock

Four requested files added as new footage posts. Existing archive reels are unchanged.

| Video | Operation | Duration | Dimensions | Frames |
| --- | --- | --- | --- | --- |
| [capricorn-sisters](../../posts/milan/caprocorn-sister-war-plague-earthquake-footage/reel.mp4) | Byte-identical copy | 30.000000 s | 760 × 1080 | 900 |
| [viperella](../../posts/milan/viperella-footage/reel.mp4) | Byte-identical copy | 60.000000 s | 760 × 1080 | 1800 |
| [board-game](../../posts/kopstal/7-fam-board-game-footage/reel.mp4) | Fixed outer-canvas rectangle | 27.433333 s | 2300 × 1300 | 823 |
| [sept-neuf](../../posts/kopstal/sept-neuf-footage/reel.mp4) | Fixed outer-canvas rectangle | 59.066667 s | 900 × 900 | 1772 |

## Stabilization

For the two requested corrections, robust edge-line fits estimate the canvas perimeter independently in every frame; the four line intersections define a perspective transform. A three-frame median rejects isolated edge-measurement excursions. Only geometry is filtered, never painting pixels. No accumulating camera trajectory is used.

Board game occupies the fixed rectangle (500, 130, 1300, 1040) inside its original 2300 × 1300 output. Sept occupies (142, 30, 616, 840) inside its original 900 × 900 output. Both outputs preserve the physical painting aspect ratios.

The supplied corner CSVs record every input quad used in the final render. Transforming these coordinates to the fixed destination is accurate to floating-point precision; this measures mapping consistency, not the independent accuracy of physical edge detection. Actual encoded before/after frames, including early low-texture canvas, hand occlusions and end frames, were visually inspected.

## Preserved material and limits

The input MERGED-20sec is actually 27.433333 seconds. KARŞIDAN-2_1 is 59.066667 seconds. Durations, all input frames, native output dimensions and 30 fps were preserved. Original audio is stream-copied and decoded SHA-256 matches exactly. FLAT_30sec and viperella were copied with matching whole-file SHA-256. All four uploaded files decode completely.

Black corner gaps, clipping, exposure changes, motion blur and changing brush strokes already in MERGED are retained. Fixed output borders do not restore missing image data or freeze the recorded painting process. No generative fill, substituted painting, colour grade, new music, subtitles, rescheduling or Instagram upload was performed.

Corrected derivatives use H.264, slow preset, CRF 17 with a 25 Mbit/s VBV ceiling, yuv420p and explicit BT.709 conversion/tags. This keeps full native frame dimensions and avoids the GitHub 100 MiB individual-file limit. The two unchanged originals were not re-encoded.

## Reproduction

Install Python, numpy, opencv-python-headless and ffmpeg. Use `replay.py` with the original input, the matching corner CSV and destination rectangle. The CSV fixes the exact recorded geometry independently of future tracker versions. SHA-256, original relative file paths, all timing checks and audio comparisons are in manifest.json.
