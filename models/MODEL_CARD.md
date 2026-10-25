# Model Card

## Identity
- Checkpoint file: `best.pt`
- File size: 5,439,834 bytes (5.19 MiB)
- SHA-256: `ab213adb5e4d8531c4f0d16718414e4e6d6505bb663ba5af55d884c167de922a`
- Task reported by local Ultralytics load: object detection
- Class names reported by local Ultralytics load: `tree_root`, `obstacle`

## Intended use
Research, demonstration, and development for sewer-inspection image analysis. The model is not established as safety-critical or suitable as the sole basis for maintenance decisions.

## Verified observations
The owner tested the checkpoint locally:
- One sample image produced a detection reported as `tree_root`.
- A second sample image produced a detection reported as `obstacle`.

These two examples confirm that inference ran; they are not a statistically meaningful evaluation.

## Limitations
- Dataset source, dataset size, and class balance are not yet documented here.
- Quantitative performance metrics have not been supplied.
- `obstacle` is a broad label and does not imply separate recognition of plastic, sludge, or other material types.
- Performance under different lighting, camera angles, pipe conditions, and image quality needs systematic testing.

## Redistribution
Before public release, confirm that the owner has the rights to redistribute the checkpoint and that any upstream model and dataset terms are satisfied.
