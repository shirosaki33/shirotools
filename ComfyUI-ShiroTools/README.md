# ComfyUI-ShiroTools v4

Adds Boundary De-Stutter Loop Cleaner (Shiro).

Recommended first test:
- low_motion_ratio = 0.60
- start_scan_frames = 12
- end_scan_frames = 12
- max_start_drop = 5
- max_end_drop = 3

This node trims only low-motion duplicate-like frames from the beginning/end of the loop, instead of trying to remove frames from the middle.


## Last-First Context Bridge (Shiro)
Uses the specific frames [prev, last, first, next] as a temporal line and inserts bridge frame(s) between last and first.
Recommended first test: bridge_frame_count=1, method=catmull_rom.


## Last-First RIFE Context / Append RIFE Bridge Frames
Builds a local context [-2, -1, 0, 1], sends it to FL_RIFE, extracts only the interpolated frame(s) between last and first, and appends them to the loop.


## v8 RIFE Local Bridge update
- Last-First RIFE Context now supports context_before/context_after.
- Recommended test: context_before=3, context_after=3, bridge_frame_count=1, rife_multiplier=4.
- Append RIFE Bridge Frames now supports pick_mode and insert_mode.
- Recommended insert test: pick_mode=balanced, insert_mode=replace_last_and_append.


## Last-First Context Bridge Advanced (Shiro)
No-model bridge node. Uses local frames around the loop boundary, estimates motion before the last frame and after the first frame, then creates bridge frame(s) using balanced, motion_extrapolated, catmull_rom, bezier, or linear math. It does not load any AI model.
