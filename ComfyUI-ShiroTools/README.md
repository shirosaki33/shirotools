# ComfyUI-ShiroTools v10

ComfyUI-ShiroTools is a small general-purpose utility node pack for ComfyUI. It is meant for practical workflow fixes rather than one specialized category.

Current areas include timing helpers, passthrough delay tools, video loop cleanup, boundary de-stutter, no-model last/first bridge generation, and local RIFE bridge helpers.

## Timing tools

### Image Workflow Delay (Shiro)
Passes an IMAGE through after a configurable delay.

### Passthrough Workflow Delay (Shiro)
Passes any ComfyUI data through after a configurable delay.

## Video loop tools

### Boundary De-Stutter Loop Cleaner (Shiro)
Trims low-motion duplicate-like frames from the beginning and/or end of a loop after interpolation.

Recommended first test:
- low_motion_ratio = 0.50 to 0.60
- start_scan_frames = 12
- end_scan_frames = 12
- max_start_drop = 5
- max_end_drop = 3 to 5

This is usually the node that removes the actual micro-stutter/travada.

### Last-First Context Bridge Advanced (Shiro)
No-model bridge node. It reads local temporal context around the loop boundary and creates bridge frame(s) between the last frame and the first frame using mathematical interpolation and motion estimation.

Useful methods:
- balanced_anti_first: default; similar to balanced, but avoids creating a bridge that is too close to frame 0.
- balanced: mixes Catmull-Rom, motion extrapolation, and linear blending.
- motion_extrapolated: estimates motion before the last frame and after the first frame.
- catmull_rom: curve interpolation using previous, last, first, and next frames.
- bezier: cubic curve using estimated motion vectors.
- linear: direct blend between last and first.

Important controls:
- max_first_progress: limits how close the bridge is allowed to drift toward frame 0.
- anti_first_strength: how strongly the node pushes a too-first-like bridge back toward the last-frame side.
- insert_mode: append, replace_last, or replace_last_and_append.

### Last-First RIFE Context (Shiro)
Builds a small local frame batch around the last→first boundary for FL_RIFE.

Example with context_before=3 and context_after=3:
- -3, -2, -1, 0, 1, 2, 3

The RIFE model only runs on this tiny boundary context instead of the whole video.

### Append RIFE Bridge Frames (Shiro)
Extracts only the interpolated RIFE candidate frame(s) between last and first and inserts them back into the original loop.

v10 adds anti-first selection logic:
- anti_first_bias: default; prefers bridge candidates that are not too similar to frame 0.
- prefer_last_side: stronger last-side bias for cases where the bridge keeps becoming almost identical to the first frame.
- balanced: evenly spaced visual progress.
- middle, first, last: simple direct candidate selection.

Important controls:
- min_distance_from_first: rejects/penalizes candidates that are nearly duplicates of frame 0.
- max_first_progress: progress limit; lower values make the bridge stay farther from frame 0.
- first_penalty_strength: penalty intensity when the candidate is too first-like.
- insert_mode: append, replace_last, or replace_last_and_append.

Recommended first test:
- pick_mode = anti_first_bias
- min_distance_from_first = 0.001
- max_first_progress = 0.72
- first_penalty_strength = 2.0
- insert_mode = replace_last_and_append
