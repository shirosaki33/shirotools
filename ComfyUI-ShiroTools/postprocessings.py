import math
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (Vídeo e Loop)
# ---------------------------------------------------------------------------

def _clamp_int(value, low, high):
    return max(low, min(high, int(value)))

def _prepare_score_features(images, max_side=96, use_luma=True):
    x = images.detach().float().cpu()
    if x.dim() != 4: raise ValueError("IMAGE batch must be a 4D tensor [N,H,W,C].")
    x = x.permute(0, 3, 1, 2).contiguous()
    n, c, h, w = x.shape
    if c > 3:
        x = x[:, :3]
        c = 3
    if use_luma and c >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
        x = (x[:, :3] * weights).sum(dim=1, keepdim=True)
    elif c != 1:
        x = x.mean(dim=1, keepdim=True)
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        nh = max(8, int(round(h * scale)))
        nw = max(8, int(round(w * scale)))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]); dx = F.pad(dx, (0, 1, 0, 0))
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]); dy = F.pad(dy, (0, 0, 0, 1))
    return x, dx + dy

def _mse(a, b):
    return float(torch.mean((a - b) ** 2).item())

def _motion_score(feat, end_idx, start_idx, window):
    n = feat.shape[0]; vals = []
    for k in range(1, int(window) + 1):
        a0 = end_idx - k; a1 = end_idx - k + 1
        b0 = start_idx + k - 1; b1 = start_idx + k
        if a0 < 0 or a1 < 0 or b0 >= n or b1 >= n: continue
        pre = feat[a1] - feat[a0]; post = feat[b1] - feat[b0]
        vals.append(_mse(pre, post))
    return sum(vals) / len(vals) if vals else 0.0

def _build_candidates(n, start_from, start_to, end_from, end_to, min_loop_frames):
    if start_to <= 0: start_to = max(0, int(round(n * 0.35)))
    if end_from <= 0: end_from = max(0, int(round(n * 0.55)))
    if end_to <= 0: end_to = n - 1
    start_from = _clamp_int(start_from, 0, n - 1)
    start_to = _clamp_int(start_to, start_from, n - 1)
    end_from = _clamp_int(end_from, 0, n - 1)
    end_to = _clamp_int(end_to, end_from, n - 1)
    min_loop_frames = max(2, int(min_loop_frames)); candidates = []
    for s in range(start_from, start_to + 1):
        min_e = max(end_from, s + min_loop_frames - 1)
        if min_e > end_to: continue
        for e in range(min_e, end_to + 1): candidates.append((s, e))
    return candidates

def _pair_image_similarity(a, b, color_compare=True):
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    if bool(color_compare):
        dims = (0, 1)
        mean_a = a.mean(dim=dims, keepdim=True); std_a = a.std(dim=dims, unbiased=False, keepdim=True).clamp(min=1e-6)
        mean_b = b.mean(dim=dims, keepdim=True); std_b = b.std(dim=dims, unbiased=False, keepdim=True).clamp(min=1e-6)
        b = ((b - mean_b) / std_b) * std_a + mean_a; b = b.clamp(0.0, 1.0)
    return float(torch.mean(torch.abs(a - b)).item())

def _shiro_green_log(message):
    try: print(f"\033[92m{message}\033[0m")
    except Exception: print(message)

def _shiro_list_input_images():
    input_dir = folder_paths.get_input_directory()
    try:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))
    except Exception: return []

def _shiro_image_tensor_to_pil(frame):
    x = frame.detach().float().cpu().clamp(0, 1)
    arr = (x.numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim != 3: raise ValueError("Expected one image frame with shape [H,W,C].")
    c = arr.shape[-1]
    if c == 1: return Image.fromarray(arr[:, :, 0], mode="L").convert("RGBA")
    if c == 3: return Image.fromarray(arr, mode="RGB").convert("RGBA")
    if c == 4: return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"Unsupported image channel count: {c}.")

def _shiro_pil_to_image_tensor(img, channels=3):
    if channels == 1: arr = np.asarray(img.convert("L"), dtype=np.float32)[:, :, None] / 255.0
    elif channels == 4: arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    else: arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).float()

def _shiro_watermark_position(position, base_w, base_h, wm_w, wm_h, padding_x, padding_y, custom_x=1.0, custom_y=1.0):
    pos = str(position or "bottom-right").lower().strip().replace("_", "-")
    aliases = {"middle": "center", "centre": "center", "top-middle": "top-center", "bottom-middle": "bottom-center", "middle-left": "center-left", "middle-right": "center-right", "left-center": "center-left", "right-center": "center-right", "center-top": "top-center", "center-bottom": "bottom-center"}
    pos = aliases.get(pos, pos)
    if pos == "custom":
        cx = max(0.0, min(1.0, float(custom_x))); cy = max(0.0, min(1.0, float(custom_y)))
        max_x = max(0, int(base_w) - int(wm_w)); max_y = max(0, int(base_h) - int(wm_h))
        return int(round(max_x * cx)), int(round(max_y * cy))
    x_positions = {"top-left": int(padding_x), "center-left": int(padding_x), "bottom-left": int(padding_x), "top-center": (base_w - wm_w) // 2, "center": (base_w - wm_w) // 2, "bottom-center": (base_w - wm_w) // 2, "top-right": base_w - wm_w - int(padding_x), "center-right": base_w - wm_w - int(padding_x), "bottom-right": base_w - wm_w - int(padding_x)}
    y_positions = {"top-left": int(padding_y), "top-center": int(padding_y), "top-right": int(padding_y), "center-left": (base_h - wm_h) // 2, "center": (base_h - wm_h) // 2, "center-right": (base_h - wm_h) // 2, "bottom-left": base_h - wm_h - int(padding_y), "bottom-center": base_h - wm_h - int(padding_y), "bottom-right": base_h - wm_h - int(padding_y)}
    if pos not in x_positions or pos not in y_positions: pos = "bottom-right"
    return x_positions[pos], y_positions[pos]

def _shiro_watermark_auto_target_width(base_w, base_h, wm_w, wm_h, scale_mode, scale, auto_scale_percent, min_size_px, max_size_px):
    mode = str(scale_mode or "manual_width").lower().strip()
    scale = max(1, min(100, int(scale))); auto_scale_percent = max(1, min(100, int(auto_scale_percent)))
    min_size_px = max(1, int(min_size_px)); max_size_px = max(min_size_px, int(max_size_px))
    if mode == "manual_width": target_w = base_w * (scale / 100.0)
    elif mode == "auto_short_side": target_w = min(base_w, base_h) * (auto_scale_percent / 100.0)
    elif mode == "auto_long_side": target_w = max(base_w, base_h) * (auto_scale_percent / 100.0)
    elif mode == "auto_area": target_w = math.sqrt(max(1, base_w * base_h)) * (auto_scale_percent / 100.0)
    else: target_w = base_w * (scale / 100.0)
    target_w = max(min_size_px, min(max_size_px, int(round(target_w))))
    target_w = max(1, min(int(target_w), int(base_w)))
    target_h = max(1, int(round(wm_h * target_w / max(1, wm_w))))
    if target_h > base_h:
        target_h = max(1, int(base_h))
        target_w = max(1, int(round(wm_w * target_h / max(1, wm_h))))
    return int(target_w), int(target_h)

def _shiro_watermark_optical_padding(position, padding_x, padding_y, wm_w, wm_h, optical_padding, optical_strength):
    if not bool(optical_padding): return int(padding_x), int(padding_y)
    strength = max(0, min(100, int(optical_strength))) / 100.0
    pos = str(position or "bottom-right").lower().strip().replace("_", "-")
    if pos in ("center", "middle", "centre"): return int(padding_x), int(padding_y)
    offset_x = min(max(0, int(padding_x)) + int(round(wm_w * 0.06 * strength)), max(0, int(padding_x)) + 96)
    offset_y = min(max(0, int(padding_y)) + int(round(wm_h * 0.06 * strength)), max(0, int(padding_y)) + 96)
    return offset_x, offset_y

# ---------------------------------------------------------------------------
# NODES (Vídeo, Loop e Watermark Simplified)
# ---------------------------------------------------------------------------

class ShiroAutoSeamLoopCut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_search_from": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "start_search_to_0_auto": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "end_search_from_0_auto": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "end_search_to_0_auto": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "min_loop_frames": ("INT", {"default": 24, "min": 2, "max": 999999, "step": 1}),
                "compare_window": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "preview_window": ("INT", {"default": 12, "min": 1, "max": 64, "step": 1}),
                "appearance_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "motion_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "edge_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05}),
                "max_eval_pairs": ("INT", {"default": 12000, "min": 100, "max": 500000, "step": 100}),
                "use_luma": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("loop_frames", "seam_preview", "best_start", "best_end", "best_score")
    FUNCTION = "find_and_cut"
    CATEGORY = "Shiro Tools/Video Management"

    def find_and_cut(self, images, start_search_from=0, start_search_to_0_auto=0, end_search_from_0_auto=0, end_search_to_0_auto=0, min_loop_frames=24, compare_window=4, preview_window=12, appearance_weight=1.0, motion_weight=2.0, edge_weight=0.5, max_eval_pairs=12000, use_luma=True, seed=0):
        if not torch.is_tensor(images) or images.dim() != 4 or images.shape[0] < 2: return (images, images, 0, max(0, int(images.shape[0]) - 1), 0.0)
        feat, edge = _prepare_score_features(images, max_side=96, use_luma=use_luma)
        n = int(feat.shape[0])
        candidates = _build_candidates(n, start_search_from, start_search_to_0_auto, end_search_from_0_auto, end_search_to_0_auto, min_loop_frames)
        if not candidates: return (images, images, 0, n - 1, 0.0)
        if len(candidates) > int(max_eval_pairs):
            step = max(1, int(math.ceil(len(candidates) / float(max_eval_pairs))))
            candidates = candidates[::step]
        best = None; best_score = None
        for s, e in candidates:
            appearance = _mse(feat[e], feat[s])
            motion = _motion_score(feat, e, s, compare_window)
            edge_delta = _mse(edge[e], edge[s])
            score = appearance_weight * appearance + motion_weight * motion + edge_weight * edge_delta
            if best_score is None or score < best_score:
                best_score = score; best = (s, e)
        if best is None: return (images, images, 0, n - 1, 0.0)
        s, e = best
        loop_frames = images[s:e + 1]
        pw = max(1, int(preview_window))
        end_preview = images[max(s, e - pw + 1):e + 1]
        start_preview = images[s:min(e + 1, s + pw)]
        seam_preview = torch.cat([end_preview, start_preview], dim=0)
        return (loop_frames, seam_preview, int(s), int(e), float(best_score))

class ShiroMotionDeStutterLoopCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "low_motion_ratio": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "protected_start_frames": ("INT", {"default": 2, "min": 0, "max": 999999, "step": 1}),
                "protected_end_frames": ("INT", {"default": 2, "min": 0, "max": 999999, "step": 1}),
                "max_frames_to_drop": ("INT", {"default": 3, "min": 0, "max": 999999, "step": 1}),
                "max_consecutive_drops": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "dropped_count", "motion_cutoff", "median_motion")
    FUNCTION = "clean"
    CATEGORY = "Shiro Tools/Video Management"

    def clean(self, images, low_motion_ratio=0.35, protected_start_frames=2, protected_end_frames=2, max_frames_to_drop=3, max_consecutive_drops=2, seed=0):
        if (not torch.is_tensor(images)) or images.dim() != 4: return (images, 0, 0.0, 0.0)
        n = int(images.shape[0])
        if n < 3: return (images, 0, 0.0, 0.0)
        x = images.detach().float().cpu()
        diffs = torch.mean(torch.abs(x[1:] - x[:-1]), dim=(1, 2, 3))
        scores = [0.0] + [float(v.item()) for v in diffs]
        protected_start_frames = _clamp_int(protected_start_frames, 0, max(0, n - 1))
        protected_end_frames = _clamp_int(protected_end_frames, 0, max(0, n - 1))
        max_frames_to_drop = max(0, int(max_frames_to_drop)); max_consecutive_drops = max(1, int(max_consecutive_drops))
        valid_start = max(1, protected_start_frames); valid_end = max(valid_start, n - protected_end_frames)
        valid_indices = list(range(valid_start, valid_end))
        if not valid_indices or max_frames_to_drop == 0:
            median_motion = float(torch.tensor(scores[1:]).median().item()) if n > 1 else 0.0
            return (images, 0, 0.0, median_motion)
        valid_vals = [scores[i] for i in valid_indices]
        median_motion = float(torch.tensor(valid_vals, dtype=torch.float32).median().item()) if valid_vals else 0.0
        motion_cutoff = float(max(0.0, median_motion * float(low_motion_ratio)))
        candidates = [i for i in valid_indices if scores[i] <= motion_cutoff]
        candidates.sort(key=lambda idx: (scores[idx], idx))
        selected = []; selected_set = set()
        for idx in candidates:
            if len(selected) >= max_frames_to_drop: break
            run = 1; left = idx - 1
            while left in selected_set: run += 1; left -= 1
            right = idx + 1
            while right in selected_set: run += 1; right += 1
            if run > max_consecutive_drops: continue
            selected.append(idx); selected_set.add(idx)
        if not selected: return (images, 0, motion_cutoff, median_motion)
        keep = [i for i in range(n) if i not in selected_set]
        if len(keep) < 2: return (images, 0, motion_cutoff, median_motion)
        cleaned = images[keep]
        return (cleaned, len(selected), motion_cutoff, median_motion)

class ShiroBoundaryDeStutterLoopCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "low_motion_ratio": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 3.0, "step": 0.05}),
                "start_scan_frames": ("INT", {"default": 12, "min": 1, "max": 999999, "step": 1}),
                "end_scan_frames": ("INT", {"default": 12, "min": 1, "max": 999999, "step": 1}),
                "max_start_drop": ("INT", {"default": 5, "min": 0, "max": 999999, "step": 1}),
                "max_end_drop": ("INT", {"default": 3, "min": 0, "max": 999999, "step": 1}),
                "min_remaining_frames": ("INT", {"default": 16, "min": 2, "max": 999999, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "start_dropped", "end_dropped", "motion_cutoff", "median_motion")
    FUNCTION = "clean"
    CATEGORY = "Shiro Tools/Video Management"

    def clean(self, images, low_motion_ratio=0.60, start_scan_frames=12, end_scan_frames=12, max_start_drop=5, max_end_drop=3, min_remaining_frames=16, seed=0):
        if (not torch.is_tensor(images)) or images.dim() != 4: return (images, 0, 0, 0.0, 0.0)
        n = int(images.shape[0])
        if n < 3: return (images, 0, 0, 0.0, 0.0)
        x = images.detach().float().cpu()
        diffs = torch.mean(torch.abs(x[1:] - x[:-1]), dim=(1, 2, 3))
        vals = [float(v.item()) for v in diffs]
        if not vals: return (images, 0, 0, 0.0, 0.0)
        median_motion = float(torch.tensor(vals, dtype=torch.float32).median().item())
        motion_cutoff = float(max(0.0, median_motion * float(low_motion_ratio)))
        start_scan_frames = max(1, int(start_scan_frames)); end_scan_frames = max(1, int(end_scan_frames))
        max_start_drop = max(0, int(max_start_drop)); max_end_drop = max(0, int(max_end_drop))
        min_remaining_frames = max(2, int(min_remaining_frames))
        start_drop = 0
        for i in range(0, min(start_scan_frames, len(vals))):
            if start_drop >= max_start_drop: break
            if n - (start_drop + 1) < min_remaining_frames: break
            if vals[i] <= motion_cutoff: start_drop += 1
            else: break
        end_drop = 0
        start_limit = max(0, len(vals) - end_scan_frames)
        for i in range(len(vals) - 1, start_limit - 1, -1):
            if end_drop >= max_end_drop: break
            if n - start_drop - (end_drop + 1) < min_remaining_frames: break
            if vals[i] <= motion_cutoff: end_drop += 1
            else: break
        if start_drop == 0 and end_drop == 0: return (images, 0, 0, motion_cutoff, median_motion)
        end_index = n - end_drop
        cleaned = images[start_drop:end_index].contiguous()
        return (cleaned, int(start_drop), int(end_drop), motion_cutoff, median_motion)

class ShiroLastFirstRIFEContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_before": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "context_after": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "bridge_frame_count": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "rife_multiplier": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("original_images", "rife_context_frames", "rife_multiplier", "bridge_frame_count", "boundary_segment_index")
    FUNCTION = "build_context"
    CATEGORY = "Shiro Tools/Video Management"

    def build_context(self, images, context_before=3, context_after=3, bridge_frame_count=1, rife_multiplier=4, seed=0):
        c = max(1, int(bridge_frame_count)); m = max(c + 1, int(rife_multiplier))
        before = max(1, min(32, int(context_before))); after = max(1, min(32, int(context_after)))
        boundary_segment_index = before - 1
        if (not torch.is_tensor(images)) or images.dim() != 4: return (images, images, int(m), int(c), int(boundary_segment_index))
        n = int(images.shape[0])
        if n <= 0: return (images, images, int(m), int(c), int(boundary_segment_index))
        frames = []
        for k in range(before, 0, -1):
            idx = max(0, n - k); frames.append(images[idx])
        for k in range(0, after + 1):
            idx = min(n - 1, k); frames.append(images[idx])
        context = torch.stack(frames, dim=0).contiguous()
        return (images.contiguous(), context, int(m), int(c), int(boundary_segment_index))

class ShiroAppendRIFEBridgeFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "rife_context_result": ("IMAGE",),
                "bridge_frame_count": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "rife_multiplier": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "boundary_segment_index": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "pick_mode": (["anti_first_bias", "balanced", "prefer_last_side", "middle", "first", "last"], {"default": "anti_first_bias"}),
                "insert_mode": (["append", "replace_last", "replace_last_and_append", "replace_first", "replace_both_sides"], {"default": "replace_last_and_append"}),
                "min_distance_from_first": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0005}),
                "max_first_progress": ("FLOAT", {"default": 0.72, "min": 0.05, "max": 0.98, "step": 0.01}),
                "first_penalty_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "bridge_frames")
    FUNCTION = "append"
    CATEGORY = "Shiro Tools/Video Management"

    def _mse_img(self, a, b): return float(torch.mean((a.float() - b.float()) ** 2).item())

    def _pick_candidates(self, candidates, original_images, count, pick_mode, min_distance_from_first=0.001, max_first_progress=0.72, first_penalty_strength=2.0):
        total = int(candidates.shape[0]); count = max(1, min(int(count), total))
        if total <= count: return candidates[:count]
        if pick_mode == "first": return candidates[:count]
        if pick_mode == "last": return candidates[-count:]
        if pick_mode == "middle":
            center = total // 2; start = max(0, min(total - count, center - (count // 2)))
            return candidates[start:start + count]
        last = original_images[-1]; first = original_images[0]
        progress = []; d_first_values = []
        for i in range(total):
            f = candidates[i]; d_last = self._mse_img(f, last); d_first = self._mse_img(f, first)
            denom = d_last + d_first
            p = (i + 1) / float(total + 1) if denom <= 1e-12 else d_last / denom
            progress.append(float(p)); d_first_values.append(float(d_first))
        max_p = max(0.05, min(0.98, float(max_first_progress)))
        min_d_first = max(0.0, float(min_distance_from_first))
        penalty_strength = max(0.0, float(first_penalty_strength))
        selected = []; used = set()
        for j in range(1, count + 1):
            base_target = j / float(count + 1)
            if pick_mode == "prefer_last_side": target = base_target * 0.70
            elif pick_mode == "anti_first_bias": target = min(base_target * 0.85, max_p)
            else: target = base_target
            best_i = None; best_score = None
            for i, p in enumerate(progress):
                if i in used: continue
                score = abs(p - target)
                if pick_mode in ("anti_first_bias", "prefer_last_side"):
                    if p > max_p: score += (p - max_p) * penalty_strength
                    if min_d_first > 0.0 and d_first_values[i] < min_d_first:
                        score += ((min_d_first - d_first_values[i]) / min_d_first) * penalty_strength
                if best_score is None or score < best_score:
                    best_score = score; best_i = i
            if best_i is not None:
                selected.append(best_i); used.add(best_i)
        if not selected: selected = [total // 2]
        return candidates[sorted(selected)]

    def append(self, original_images, rife_context_result, bridge_frame_count=1, rife_multiplier=4, boundary_segment_index=2, pick_mode="anti_first_bias", insert_mode="replace_last_and_append", min_distance_from_first=0.001, max_first_progress=0.72, first_penalty_strength=2.0, seed=0):
        if (not torch.is_tensor(original_images)) or original_images.dim() != 4:
            empty = original_images[:0] if torch.is_tensor(original_images) and original_images.dim() == 4 else original_images
            return (original_images, empty)
        count = max(1, int(bridge_frame_count)); m = max(count + 1, int(rife_multiplier)); seg = max(0, int(boundary_segment_index))
        if not torch.is_tensor(rife_context_result) or rife_context_result.dim() != 4: return (original_images.contiguous(), original_images[:0].contiguous())
        n = int(rife_context_result.shape[0])
        start = seg * m + 1; end = (seg + 1) * m
        if n >= end and start < end: candidates = rife_context_result[start:end]
        else:
            center = n // 2; window = max(count, m - 1)
            s = max(0, center - window // 2); e = min(n, s + window); s = max(0, e - window)
            candidates = rife_context_result[s:e]
        if candidates.shape[0] <= 0: return (original_images.contiguous(), original_images[:0].contiguous())
        bridge = self._pick_candidates(candidates, original_images, count, pick_mode, min_distance_from_first=min_distance_from_first, max_first_progress=max_first_progress, first_penalty_strength=first_penalty_strength).to(dtype=original_images.dtype).contiguous()
        original_n = int(original_images.shape[0])
        if insert_mode == "append": out = torch.cat((original_images, bridge), dim=0).contiguous()
        elif insert_mode == "replace_last":
            out = torch.cat((original_images[:-1], bridge[:1]), dim=0).contiguous(); bridge = bridge[:1].contiguous()
        elif insert_mode == "replace_last_and_append": out = torch.cat((original_images[:-1], bridge), dim=0).contiguous()
        elif insert_mode == "replace_first":
            if original_n <= 2: out = torch.cat((original_images, bridge), dim=0).contiguous()
            else: out = torch.cat((original_images[1:], bridge), dim=0).contiguous()
        else:
            if original_n <= 3: out = torch.cat((original_images[:-1], bridge), dim=0).contiguous()
            else: out = torch.cat((original_images[1:-1], bridge), dim=0).contiguous()
        return (out, bridge)

class ShiroPreInterpolationLoopBoundaryCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "compare_pairs": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
                "remove_end_frames": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "remove_start_frames": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "color_compare": ("BOOLEAN", {"default": True}),
                "image_similarity": ("FLOAT", {"default": 0.018, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_remaining_frames": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "end_frames_removed", "start_frames_removed", "closest_pair_similarity")
    FUNCTION = "clean"
    CATEGORY = "Shiro Tools/Video Management"

    def _candidate_scores(self, images, start_remove, end_remove, compare_pairs, color_compare):
        n = int(images.shape[0]); requested_pairs = max(1, int(compare_pairs)); scores = []
        for pair_idx in range(requested_pairs):
            end_idx = n - 1 - int(end_remove) - pair_idx
            start_idx = int(start_remove) + pair_idx
            if end_idx < 0 or start_idx >= n or end_idx <= start_idx: break
            scores.append(_pair_image_similarity(images[end_idx], images[start_idx], color_compare=color_compare))
        return scores

    def clean(self, images, compare_pairs=2, remove_end_frames=2, remove_start_frames=0, color_compare=True, image_similarity=0.018, min_remaining_frames=8, seed=0):
        if (not torch.is_tensor(images)) or images.dim() != 4: return (images, 0, 0, 0.0)
        n = int(images.shape[0])
        if n < 2: return (images, 0, 0, 0.0)
        compare_pairs = max(1, int(compare_pairs)); remove_end_frames = max(0, int(remove_end_frames))
        remove_start_frames = max(0, int(remove_start_frames)); min_remaining_frames = max(2, min(int(min_remaining_frames), n))
        threshold = float(image_similarity)
        valid_candidates = []; fallback_candidates = []
        max_pairs_used = 0; chosen_pair_count = 0
        for end_remove in range(remove_end_frames + 1):
            for start_remove in range(remove_start_frames + 1):
                remaining = n - end_remove - start_remove
                if remaining < min_remaining_frames: continue
                scores = self._candidate_scores(images, start_remove, end_remove, compare_pairs, color_compare)
                if not scores: continue
                pair_count = len(scores)
                max_pairs_used = max(max_pairs_used, pair_count)
                closest = min(scores); average = sum(scores) / pair_count
                all_safe = all(score > threshold for score in scores)
                total_removed = start_remove + end_remove
                key_valid = (total_removed, start_remove, -closest, -average)
                key_fallback = (-closest, -average, total_removed, start_remove)
                entry = {"start_remove": start_remove, "end_remove": end_remove, "closest": closest, "average": average, "scores": scores, "pair_count": pair_count, "key_valid": key_valid, "key_fallback": key_fallback}
                if all_safe: valid_candidates.append(entry)
                fallback_candidates.append(entry)
        if valid_candidates: chosen = min(valid_candidates, key=lambda x: x["key_valid"])
        elif fallback_candidates: chosen = min(fallback_candidates, key=lambda x: x["key_fallback"])
        else: return (images, 0, 0, 0.0)
        start_remove = int(chosen["start_remove"]); end_remove = int(chosen["end_remove"])
        if end_remove > 0: cleaned = images[start_remove:n - end_remove]
        else: cleaned = images[start_remove:]
        return (cleaned.contiguous(), end_remove, start_remove, float(chosen["closest"]))

class ShiroColorBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["auto", "video", "loop"], {"default": "loop"}),
                "reference_mode": (["batch_median", "first_frame", "reference_image"], {"default": "batch_median"}),
                "match_method": (["mean_std", "brightness", "hybrid"], {"default": "hybrid"}),
                "global_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seam_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seam_frame_count": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "affect_start_frames": ("BOOLEAN", {"default": False}),
                "affect_end_frames": ("BOOLEAN", {"default": True}),
                "auto_loop_threshold": ("FLOAT", {"default": 0.060, "min": 0.0, "max": 1.0, "step": 0.001}),
                "protect_highlights": ("BOOLEAN", {"default": True}),
                "luma_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.001}),
                "temporal_smoothing": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "highlight_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.001}),
                "tone_range_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_luma_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_grid_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "local_max_gain": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 4.0, "step": 0.001}),
                "local_temporal_radius": ("INT", {"default": 3, "min": 0, "max": 16, "step": 1}),
                "local_reference_blend": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_detail_protection": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_lowpass_passes": ("INT", {"default": 3, "min": 0, "max": 16, "step": 1}),
                "local_chroma_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_chroma_max_shift": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.001}),
                "local_chroma_detail_protection": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_chroma_threshold": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.10, "step": 0.001}),
                "local_pulse_strength": ("FLOAT", {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_pulse_radius": ("INT", {"default": 2, "min": 1, "max": 12, "step": 1}),
                "local_pulse_luma_max_shift": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 0.20, "step": 0.001}),
                "local_pulse_chroma_max_shift": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.20, "step": 0.001}),
                "local_pulse_threshold": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.05, "step": 0.001}),
                "local_pulse_detail_protection": ("FLOAT", {"default": 0.78, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_pulse_motion_protection": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_static_strength": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_static_radius": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "local_static_luma_max_shift": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.10, "step": 0.001}),
                "local_static_chroma_max_shift": ("FLOAT", {"default": 0.014, "min": 0.0, "max": 0.10, "step": 0.001}),
                "local_static_motion_threshold": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.10, "step": 0.001}),
                "local_static_neighbor_bias": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.001}),
                "local_static_detail_protection": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "detected_mode (loop/video)", "seam_delta", "global_delta")
    FUNCTION = "stabilize"
    CATEGORY = "Shiro Tools/Video Management"

    @staticmethod
    def _safe_rgb(images):
        if not torch.is_tensor(images): raise ValueError("images must be a ComfyUI IMAGE tensor.")
        single = False
        if images.dim() == 3:
            images = images.unsqueeze(0); single = True
        if images.dim() != 4: raise ValueError(f"Expected IMAGE shape [B,H,W,C] or [H,W,C], got {tuple(images.shape)}.")
        x = images.detach().float().clone(); alpha = None
        if x.shape[-1] >= 4:
            alpha = x[..., 3:4].clone(); rgb = x[..., :3].clone()
        elif x.shape[-1] == 1: rgb = x.repeat(1, 1, 1, 3)
        else: rgb = x[..., :3].clone()
        return x, rgb.clamp(0.0, 1.0), alpha, single

    @staticmethod
    def _stats(x):
        dims = (0, 1) if x.dim() == 3 else (1, 2)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True).clamp_min(1e-4)
        return mean, std

    @staticmethod
    def _luma(x):
        w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(*([1] * (x.dim() - 1)), 3)
        return (x[..., :3] * w).sum(dim=-1, keepdim=True)

    @staticmethod
    def _frame_luma_means(x):
        if x.dim() == 3: return ShiroColorBalance._luma(x).mean().view(1)
        return ShiroColorBalance._luma(x).mean(dim=(1, 2, 3))

    @staticmethod
    def _rolling_mean(values, radius, circular=False):
        radius = int(radius)
        if radius <= 0 or values.numel() <= 1: return values.clone()
        n = int(values.numel())
        if circular: radius = min(radius, max(1, n - 1))
        x = values.float().view(1, 1, n)
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device)
        if circular:
            xp = torch.cat([x[:, :, -radius:], x, x[:, :, :radius]], dim=2)
            out = F.conv1d(xp, kernel / k, padding=0).view(n)
        else:
            xp = F.pad(x, (radius, radius))
            window_sum = F.conv1d(xp, kernel, padding=0).view(n)
            positions = torch.arange(n, dtype=torch.float32, device=values.device)
            counts = ((positions + radius).clamp(max=n - 1) - (positions - radius).clamp(min=0) + 1)
            out = window_sum / counts
        return out.to(values.dtype)

    @staticmethod
    def _quantiles_per_frame(luma):
        flat = luma.reshape(luma.shape[0], -1)
        qs = torch.quantile(flat, torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=flat.dtype, device=flat.device), dim=1)
        return qs[0], qs[1], qs[2], qs[3]

    @staticmethod
    def _edge_similarity_delta(a, b):
        aa = a.detach().float()[..., :3]; bb = b.detach().float()[..., :3]
        if aa.dim() == 3:
            aa = aa.unsqueeze(0); bb = bb.unsqueeze(0)
        aa = aa.permute(0, 3, 1, 2); bb = bb.permute(0, 3, 1, 2)
        h, w = aa.shape[-2:]
        longest = max(h, w)
        if longest > 96:
            scale = 96.0 / float(longest)
            size = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))
            aa = F.interpolate(aa, size=size, mode="bilinear", align_corners=False)
            bb = F.interpolate(bb, size=size, mode="bilinear", align_corners=False)
        return float(torch.mean(torch.abs(aa - bb)).item())

    @staticmethod
    def _blend_correction(original, corrected, strength, protect_highlights):
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0: return original
        if bool(protect_highlights):
            luma = ShiroColorBalance._luma(original)
            protection = (1.0 - torch.clamp((luma - 0.82) / 0.18, 0.0, 1.0) * 0.55)
            return original * (1.0 - strength * protection) + corrected * (strength * protection)
        return original.lerp(corrected, strength)

    @staticmethod
    def _match_frame(frame, ref_mean, ref_std, method):
        mean, std = ShiroColorBalance._stats(frame)
        if method == "brightness": return frame + (ref_mean - mean)
        if method == "mean_std": return (frame - mean) * (ref_std / std) + ref_mean
        return (frame - mean) * torch.sqrt(ref_std / std) + ref_mean

    @staticmethod
    def _apply_tone_locks(out, detected_mode, luma_strength, temporal_smoothing, highlight_strength, tone_range_strength, protect_highlights):
        n = int(out.shape[0])
        if n <= 1: return out
        luma = ShiroColorBalance._luma(out)
        q50, q90, q95, q99 = ShiroColorBalance._quantiles_per_frame(luma)
        circular = detected_mode == "loop"
        radius = max(0, int(temporal_smoothing))
        target50 = ShiroColorBalance._rolling_mean(q50, radius, circular).lerp(torch.full_like(q50, q50.median()), 0.35)
        target90 = ShiroColorBalance._rolling_mean(q90, radius, circular).lerp(torch.full_like(q90, q90.median()), 0.55)
        target95 = ShiroColorBalance._rolling_mean(q95, radius, circular).lerp(torch.full_like(q95, q95.median()), 0.70)
        target99 = ShiroColorBalance._rolling_mean(q99, radius, circular).lerp(torch.full_like(q99, q99.median()), 0.85)
        luma_strength = max(0.0, min(1.0, float(luma_strength)))
        highlight_strength = max(0.0, min(1.0, float(highlight_strength)))
        tone_range_strength = max(0.0, min(1.0, float(tone_range_strength)))
        for i in range(n):
            cur50 = float(q50[i].item()); cur90 = float(q90[i].item())
            cur95 = float(q95[i].item()); cur99 = float(q99[i].item())
            if luma_strength > 0 and cur50 > 1e-5:
                gain50 = max(min(float(target50[i].item()) / cur50, 1.35), 1.0 / 1.35)
                corrected = (out[i] * gain50).clamp(0.0, 1.0)
                out[i] = ShiroColorBalance._blend_correction(out[i], corrected, luma_strength, protect_highlights)
            if tone_range_strength > 0 and cur90 > 1e-5:
                gain90 = max(min(float(target90[i].item()) / cur90, 1.35), 1.0 / 1.35)
                li = ShiroColorBalance._luma(out[i])
                w90 = torch.clamp((li - cur50) / max(cur90 - cur50, 1e-4), 0.0, 1.0)
                corrected = (out[i] * (1.0 + (gain90 - 1.0) * w90)).clamp(0.0, 1.0)
                out[i] = ShiroColorBalance._blend_correction(out[i], corrected, tone_range_strength, protect_highlights)
            if highlight_strength > 0 and cur95 > 1e-5:
                tgt = 0.65 * float(target95[i].item()) + 0.35 * float(target99[i].item())
                cur = 0.65 * cur95 + 0.35 * max(cur99, 1e-5)
                gain_hi = max(min(tgt / max(cur, 1e-5), 1.45), 1.0 / 1.45)
                li = ShiroColorBalance._luma(out[i])
                wh = torch.clamp((li - cur90) / max(cur99 - cur90, 1e-4), 0.0, 1.0)
                wh = wh * wh
                corrected = (out[i] * (1.0 + (gain_hi - 1.0) * wh)).clamp(0.0, 1.0)
                out[i] = ShiroColorBalance._blend_correction(out[i], corrected, highlight_strength, protect_highlights)
        return out

    @staticmethod
    def _rolling_mean_grid(values, radius, circular=False):
        radius = int(radius)
        if radius <= 0 or values.shape[0] <= 1: return values.clone()
        b, c, h, w = values.shape
        if circular: radius = min(radius, max(1, b - 1))
        k = 2 * radius + 1
        x = values.permute(2, 3, 1, 0).reshape(-1, 1, b).float()
        kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device)
        if circular:
            xp = torch.cat([x[:, :, -radius:], x, x[:, :, :radius]], dim=2)
            out = F.conv1d(xp, kernel / k, padding=0)
        else:
            xp = F.pad(x, (radius, radius))
            window_sum = F.conv1d(xp, kernel, padding=0)
            positions = torch.arange(b, dtype=torch.float32, device=values.device)
            counts = ((positions + radius).clamp(max=b - 1) - (positions - radius).clamp(min=0) + 1)
            out = window_sum / counts.view(1, 1, b)
        return out.reshape(h, w, c, b).permute(3, 2, 0, 1).to(values.dtype)

    @staticmethod
    def _rolling_neighbor_mean_grid(values, radius, circular=False):
        radius = max(1, int(radius))
        b, c, h, w = values.shape
        if b <= 2: return values.clone()
        if circular: radius = min(radius, max(1, b - 1))
        k = 2 * radius + 1
        x = values.permute(2, 3, 1, 0).reshape(-1, 1, b).float()
        kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device)
        if circular:
            xp = torch.cat([x[:, :, -radius:], x, x[:, :, :radius]], dim=2)
            window_sum = F.conv1d(xp, kernel, padding=0)
        else:
            window_sum = F.conv1d(x, kernel, padding=radius)
        positions = torch.arange(b, dtype=torch.float32, device=values.device)
        if circular: count = torch.full((b,), k - 1, dtype=torch.float32, device=values.device)
        else: count = ((positions + radius).clamp(max=b - 1) - (positions - radius).clamp(min=0) + 1 - 1).clamp(min=1)
        neighbor_sum = window_sum - x
        neighbor_mean = neighbor_sum / count.view(1, 1, b)
        return neighbor_mean.reshape(h, w, c, b).permute(3, 2, 0, 1).to(values.dtype)

    @staticmethod
    def _neighbor_motion_grid(values, radius, circular=False):
        radius = max(1, int(radius))
        b, c, h, w = values.shape
        if b <= 2: return torch.zeros_like(values[:, :1])
        if circular: radius = min(radius, max(1, b - 1))
        x = values.permute(2, 3, 1, 0).reshape(-1, 1, b).float()
        kernel = torch.ones(1, 1, radius, dtype=x.dtype, device=x.device) / radius
        if circular:
            xp_prev = torch.cat([x[:, :, b - radius:], x], dim=2)
            prev = F.conv1d(xp_prev, kernel, padding=0)[:, :, :b]
            xp_next = torch.cat([x[:, :, 1:], x[:, :, :radius]], dim=2)
            next_ = F.conv1d(xp_next, kernel, padding=0)[:, :, :b]
            motion = torch.abs(prev - next_)
        else:
            positions = torch.arange(b, dtype=torch.float32, device=values.device)
            n_prev = (positions).clamp(max=radius)
            n_next = (b - 1 - positions).clamp(max=radius)
            xp_prev = F.pad(x, (radius, 0))
            prev_sum = F.conv1d(xp_prev, torch.ones_like(kernel), padding=0)[:, :, :b]
            prev_cnt = n_prev.clamp(min=1).view(1, 1, b)
            prev = prev_sum / prev_cnt
            xp_next = F.pad(x, (0, radius))
            next_sum = F.conv1d(xp_next, torch.ones_like(kernel), padding=0)[:, :, 1:b + 1]
            next_cnt = n_next.clamp(min=1).view(1, 1, b)
            next_ = next_sum / next_cnt
            has_prev = (n_prev > 0).float().view(1, 1, b)
            has_next = (n_next > 0).float().view(1, 1, b)
            motion = torch.abs(prev - next_) * has_prev * has_next
        return motion.reshape(h, w, c, b).mean(dim=2, keepdim=True).permute(3, 2, 0, 1).to(values.dtype)

    @staticmethod
    def _smooth_gain_grid(gain, passes):
        passes = max(0, int(passes))
        if passes <= 0 or gain.shape[-2] < 3 or gain.shape[-1] < 3: return gain
        out = gain
        for _ in range(passes):
            padded = F.pad(out, (1, 1, 1, 1), mode="replicate")
            out = F.avg_pool2d(padded, kernel_size=3, stride=1, padding=0)
        return out

    @staticmethod
    def _apply_local_luma_lock(out, detected_mode, strength, grid_size, max_gain, temporal_radius=3, reference_blend=0.85, detail_protection=0.25, lowpass_passes=2):
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0 or out.shape[0] <= 1: return out
        grid_size = max(8, int(grid_size)); max_gain = max(1.0, float(max_gain))
        temporal_radius = max(0, int(temporal_radius)); reference_blend = max(0.0, min(1.0, float(reference_blend)))
        detail_protection = max(0.0, min(1.0, float(detail_protection))); lowpass_passes = max(0, int(lowpass_passes))
        n, h, w, c = out.shape
        eps = 1e-5
        luma = ShiroColorBalance._luma(out).permute(0, 3, 1, 2).clamp_min(eps)
        gh = max(1, int(round(h / grid_size))); gw = max(1, int(round(w / grid_size)))
        local = F.adaptive_avg_pool2d(luma, (gh, gw)).clamp_min(eps)
        local_sq = F.adaptive_avg_pool2d(luma * luma, (gh, gw))
        local_std = torch.sqrt(torch.clamp(local_sq - local * local, min=0.0))
        circular = detected_mode == "loop"
        median_ref = local.median(dim=0, keepdim=True).values.expand_as(local)
        temporal_ref = ShiroColorBalance._rolling_mean_grid(local, temporal_radius, circular=circular)
        target = temporal_ref.lerp(median_ref, reference_blend).clamp_min(eps)
        log_limit = math.log(max_gain)
        log_gain = (torch.log(target) - torch.log(local)).clamp(-log_limit, log_limit)
        if detail_protection > 0.0:
            rel_detail = (local_std / local).clamp(0.0, 2.0)
            detail_mask = torch.clamp((rel_detail - 0.06) / 0.22, 0.0, 1.0)
            log_gain = log_gain * (1.0 - detail_protection * detail_mask)
        gain = torch.exp(log_gain).clamp(1.0 / max_gain, max_gain)
        gain = ShiroColorBalance._smooth_gain_grid(gain, lowpass_passes)
        gain_full = F.interpolate(gain, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        corrected = (out * gain_full).clamp(0.0, 1.0)
        return out.lerp(corrected, strength).clamp(0.0, 1.0)

    @staticmethod
    def _apply_local_chroma_lock(out, detected_mode, strength, grid_size, max_shift, temporal_radius=3, reference_blend=0.85, detail_protection=0.65, lowpass_passes=2, threshold=0.018):
        strength = max(0.0, min(1.0, float(strength))); max_shift = max(0.0, float(max_shift))
        if strength <= 0.0 or max_shift <= 0.0 or out.shape[0] <= 1: return out
        grid_size = max(8, int(grid_size)); temporal_radius = max(0, int(temporal_radius))
        reference_blend = max(0.0, min(1.0, float(reference_blend))); detail_protection = max(0.0, min(1.0, float(detail_protection)))
        lowpass_passes = max(0, int(lowpass_passes)); threshold = max(0.0, float(threshold))
        n, h, w, c = out.shape
        rgb = out[..., :3].permute(0, 3, 1, 2).contiguous()
        luma = ShiroColorBalance._luma(out).permute(0, 3, 1, 2).contiguous()
        gh = max(1, int(round(h / grid_size))); gw = max(1, int(round(w / grid_size)))
        local_rgb = F.adaptive_avg_pool2d(rgb, (gh, gw))
        local_luma = F.adaptive_avg_pool2d(luma, (gh, gw))
        local_chroma = local_rgb - local_luma
        local_luma_sq = F.adaptive_avg_pool2d(luma * luma, (gh, gw))
        local_std = torch.sqrt(torch.clamp(local_luma_sq - local_luma * local_luma, min=0.0))
        rel_detail = (local_std / local_luma.clamp_min(1e-5)).clamp(0.0, 2.0)
        circular = detected_mode == "loop"
        median_ref = local_chroma.median(dim=0, keepdim=True).values.expand_as(local_chroma)
        temporal_ref = ShiroColorBalance._rolling_mean_grid(local_chroma, temporal_radius, circular=circular)
        chroma_reference_blend = max(0.0, min(0.35, reference_blend * 0.35))
        target = temporal_ref.lerp(median_ref, chroma_reference_blend)
        delta = (target - local_chroma).clamp(-max_shift, max_shift)
        if threshold > 0.0:
            denom = max(max_shift - threshold, 1e-5)
            mask = torch.clamp((delta.abs() - threshold) / denom, 0.0, 1.0)
            delta = delta * mask
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=delta.dtype, device=delta.device).view(1, 3, 1, 1)
        delta_luma = (delta * weights).sum(dim=1, keepdim=True)
        delta = delta - delta_luma
        if detail_protection > 0.0:
            detail_mask = torch.clamp((rel_detail - 0.05) / 0.24, 0.0, 1.0)
            delta = delta * (1.0 - detail_protection * detail_mask)
        delta = ShiroColorBalance._smooth_gain_grid(delta, lowpass_passes)
        delta_full = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        corrected = (out[..., :3] + delta_full).clamp(0.0, 1.0)
        out2 = out.clone()
        out2[..., :3] = out[..., :3].lerp(corrected, strength)
        return out2.clamp(0.0, 1.0)

    @staticmethod
    def _apply_local_residual_pulse_lock(out, detected_mode, strength, grid_size, radius=2, luma_max_shift=0.030, chroma_max_shift=0.020, threshold=0.006, detail_protection=0.70, lowpass_passes=2, motion_protection=0.65):
        strength = max(0.0, min(1.0, float(strength))); luma_max_shift = max(0.0, float(luma_max_shift))
        chroma_max_shift = max(0.0, float(chroma_max_shift))
        if strength <= 0.0 or out.shape[0] <= 2 or (luma_max_shift <= 0.0 and chroma_max_shift <= 0.0): return out
        grid_size = max(8, int(grid_size)); radius = max(1, int(radius)); threshold = max(0.0, float(threshold))
        detail_protection = max(0.0, min(1.0, float(detail_protection))); motion_protection = max(0.0, min(1.0, float(motion_protection)))
        lowpass_passes = max(0, int(lowpass_passes))
        n, h, w, c = out.shape
        rgb = out[..., :3].permute(0, 3, 1, 2).contiguous()
        luma = ShiroColorBalance._luma(out).permute(0, 3, 1, 2).contiguous()
        gh = max(1, int(round(h / grid_size))); gw = max(1, int(round(w / grid_size)))
        local_rgb = F.adaptive_avg_pool2d(rgb, (gh, gw))
        local_luma = F.adaptive_avg_pool2d(luma, (gh, gw))
        local_chroma = local_rgb - local_luma
        circular = detected_mode == "loop"
        baseline_luma = ShiroColorBalance._rolling_neighbor_mean_grid(local_luma, radius, circular=circular)
        baseline_chroma = ShiroColorBalance._rolling_neighbor_mean_grid(local_chroma, radius, circular=circular)
        delta_luma = (baseline_luma - local_luma).clamp(-luma_max_shift, luma_max_shift)
        delta_chroma = (baseline_chroma - local_chroma).clamp(-chroma_max_shift, chroma_max_shift)
        if threshold > 0.0:
            if luma_max_shift > 0.0:
                denom_l = max(luma_max_shift - threshold, 1e-5)
                mask_l = torch.clamp((delta_luma.abs() - threshold) / denom_l, 0.0, 1.0)
                delta_luma = delta_luma * mask_l
            if chroma_max_shift > 0.0:
                denom_c = max(chroma_max_shift - threshold, 1e-5)
                mask_c = torch.clamp((delta_chroma.abs() - threshold) / denom_c, 0.0, 1.0)
                delta_chroma = delta_chroma * mask_c
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=delta_chroma.dtype, device=delta_chroma.device).view(1, 3, 1, 1)
        delta_chroma = delta_chroma - (delta_chroma * weights).sum(dim=1, keepdim=True)
        if detail_protection > 0.0:
            local_luma_sq = F.adaptive_avg_pool2d(luma * luma, (gh, gw))
            local_std = torch.sqrt(torch.clamp(local_luma_sq - local_luma * local_luma, min=0.0))
            rel_detail = (local_std / local_luma.clamp_min(1e-5)).clamp(0.0, 2.0)
            detail_mask = torch.clamp((rel_detail - 0.045) / 0.22, 0.0, 1.0)
            keep = 1.0 - detail_protection * detail_mask
            delta_luma = delta_luma * keep; delta_chroma = delta_chroma * keep
        if motion_protection > 0.0:
            motion_luma = ShiroColorBalance._neighbor_motion_grid(local_luma, radius, circular=circular)
            motion_chroma = ShiroColorBalance._neighbor_motion_grid(local_chroma, radius, circular=circular)
            keep_l = 1.0 - motion_protection * torch.clamp((motion_luma - 0.006) / 0.030, 0.0, 1.0)
            keep_c = 1.0 - motion_protection * torch.clamp((motion_chroma - 0.004) / 0.026, 0.0, 1.0)
            delta_luma = delta_luma * keep_l; delta_chroma = delta_chroma * keep_c
        if lowpass_passes > 0:
            delta_luma = ShiroColorBalance._smooth_gain_grid(delta_luma, lowpass_passes)
            delta_chroma = ShiroColorBalance._smooth_gain_grid(delta_chroma, lowpass_passes)
        delta_luma_full = F.interpolate(delta_luma, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        delta_chroma_full = F.interpolate(delta_chroma, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        corrected = (out[..., :3] + delta_luma_full + delta_chroma_full).clamp(0.0, 1.0)
        out2 = out.clone()
        out2[..., :3] = out[..., :3].lerp(corrected, strength)
        return out2.clamp(0.0, 1.0)

    @staticmethod
    def _apply_static_background_lock(out, detected_mode, strength, grid_size, radius=4, luma_max_shift=0.018, chroma_max_shift=0.012, motion_threshold=0.020, lowpass_passes=3, neighbor_bias=0.75, detail_protection=0.18):
        strength = max(0.0, min(1.0, float(strength))); luma_max_shift = max(0.0, float(luma_max_shift))
        chroma_max_shift = max(0.0, float(chroma_max_shift))
        if strength <= 0.0 or out.shape[0] <= 2 or (luma_max_shift <= 0.0 and chroma_max_shift <= 0.0): return out
        grid_size = max(8, int(grid_size)); radius = max(1, int(radius))
        motion_threshold = max(1e-5, float(motion_threshold)); lowpass_passes = max(0, int(lowpass_passes))
        neighbor_bias = max(0.0, min(1.0, float(neighbor_bias))); detail_protection = max(0.0, min(1.0, float(detail_protection)))
        n, h, w, c = out.shape
        rgb = out[..., :3].permute(0, 3, 1, 2).contiguous()
        luma = ShiroColorBalance._luma(out).permute(0, 3, 1, 2).contiguous()
        gh = max(1, int(round(h / grid_size))); gw = max(1, int(round(w / grid_size)))
        local_rgb = F.adaptive_avg_pool2d(rgb, (gh, gw))
        local_luma = F.adaptive_avg_pool2d(luma, (gh, gw))
        local_chroma = local_rgb - local_luma
        circular = detected_mode == "loop"
        slow_luma = ShiroColorBalance._rolling_mean_grid(local_luma, radius, circular=circular)
        slow_chroma = ShiroColorBalance._rolling_mean_grid(local_chroma, radius, circular=circular)
        neighbor_luma = ShiroColorBalance._rolling_neighbor_mean_grid(local_luma, max(1, min(radius, 3)), circular=circular)
        neighbor_chroma = ShiroColorBalance._rolling_neighbor_mean_grid(local_chroma, max(1, min(radius, 3)), circular=circular)
        base_luma = slow_luma.lerp(neighbor_luma, neighbor_bias)
        base_chroma = slow_chroma.lerp(neighbor_chroma, neighbor_bias)
        delta_luma = (base_luma - local_luma).clamp(-luma_max_shift, luma_max_shift)
        delta_chroma = (base_chroma - local_chroma).clamp(-chroma_max_shift, chroma_max_shift)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=delta_chroma.dtype, device=delta_chroma.device).view(1, 3, 1, 1)
        delta_chroma = delta_chroma - (delta_chroma * weights).sum(dim=1, keepdim=True)
        motion_luma = ShiroColorBalance._neighbor_motion_grid(local_luma, 1, circular=circular)
        motion_chroma = ShiroColorBalance._neighbor_motion_grid(local_chroma, 1, circular=circular).mean(dim=1, keepdim=True)
        motion = motion_luma + motion_chroma
        static_keep = 1.0 - torch.clamp(motion / motion_threshold, 0.0, 1.0)
        static_keep = static_keep * static_keep
        local_luma_sq = F.adaptive_avg_pool2d(luma * luma, (gh, gw))
        local_std = torch.sqrt(torch.clamp(local_luma_sq - local_luma * local_luma, min=0.0))
        rel_detail = (local_std / local_luma.clamp_min(1e-5)).clamp(0.0, 2.0)
        texture_mask = torch.clamp((rel_detail - 0.10) / 0.35, 0.0, 1.0)
        texture_keep = 1.0 - detail_protection * texture_mask
        keep = static_keep * texture_keep
        delta_luma = delta_luma * keep; delta_chroma = delta_chroma * keep
        if lowpass_passes > 0:
            delta_luma = ShiroColorBalance._smooth_gain_grid(delta_luma, lowpass_passes)
            delta_chroma = ShiroColorBalance._smooth_gain_grid(delta_chroma, lowpass_passes)
        delta_luma_full = F.interpolate(delta_luma, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        delta_chroma_full = F.interpolate(delta_chroma, size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        corrected = (out[..., :3] + delta_luma_full + delta_chroma_full).clamp(0.0, 1.0)
        out2 = out.clone()
        out2[..., :3] = out[..., :3].lerp(corrected, strength)
        return out2.clamp(0.0, 1.0)

    def stabilize(
        self, images, mode="loop", reference_mode="batch_median", match_method="hybrid",
        global_strength=0.18, seam_strength=0.55, seam_frame_count=8, affect_start_frames=False, affect_end_frames=True, auto_loop_threshold=0.060,
        protect_highlights=True, luma_strength=0.75, temporal_smoothing=1, highlight_strength=0.75, tone_range_strength=0.75,
        local_luma_strength=0.55, local_grid_size=32, local_max_gain=1.25, local_temporal_radius=3, local_reference_blend=0.35, local_detail_protection=0.50, local_lowpass_passes=3,
        local_chroma_strength=0.18, local_chroma_max_shift=0.025, local_chroma_detail_protection=0.75, local_chroma_threshold=0.020,
        local_pulse_strength=0.58, local_pulse_radius=2, local_pulse_luma_max_shift=0.035, local_pulse_chroma_max_shift=0.025, local_pulse_threshold=0.004, local_pulse_detail_protection=0.78, local_pulse_motion_protection=0.65,
        local_static_strength=0.38, local_static_radius=4, local_static_luma_max_shift=0.020, local_static_chroma_max_shift=0.014, local_static_motion_threshold=0.020, local_static_neighbor_bias=0.75, local_static_detail_protection=0.18,
        reference_image=None, seed=0
    ):
        original, rgb, alpha, single = self._safe_rgb(images)
        n = int(rgb.shape[0])
        if n <= 0: return (images, "empty", 0.0, 0.0)
        mode = str(mode or "auto").lower().strip()
        reference_mode = str(reference_mode or "batch_median").lower().strip()
        match_method = str(match_method or "hybrid").lower().strip()
        if match_method not in ("mean_std", "brightness", "hybrid"): match_method = "hybrid"
        seam_delta = self._edge_similarity_delta(rgb[-1], rgb[0]) if n >= 2 else 0.0
        detected_mode = mode
        if mode == "auto": detected_mode = "loop" if (n >= 3 and seam_delta <= float(auto_loop_threshold)) else "video"
        frame_mean, frame_std = self._stats(rgb)
        if reference_mode == "reference_image" and reference_image is not None:
            _, ref_rgb, _, _ = self._safe_rgb(reference_image)
            ref_mean, ref_std = self._stats(ref_rgb[:1])
        elif reference_mode == "first_frame": ref_mean, ref_std = self._stats(rgb[:1])
        else:
            ref_mean = frame_mean.median(dim=0, keepdim=True).values; ref_std = frame_std.median(dim=0, keepdim=True).values
        out = rgb.clone()
        global_strength = max(0.0, min(1.0, float(global_strength)))
        global_delta_vals = []
        if global_strength > 0.0:
            if match_method == "brightness": corrected = rgb + (ref_mean - frame_mean)
            elif match_method == "mean_std": corrected = (rgb - frame_mean) * (ref_std / frame_std) + ref_mean
            else: corrected = (rgb - frame_mean) * torch.sqrt(ref_std / frame_std) + ref_mean
            out = self._blend_correction(rgb, corrected, global_strength, protect_highlights)
            global_delta_vals = torch.mean(torch.abs(corrected - rgb), dim=(1, 2, 3)).detach().cpu().tolist()
        out = self._apply_tone_locks(out, detected_mode, luma_strength, temporal_smoothing, highlight_strength, tone_range_strength, protect_highlights)
        out = self._apply_local_luma_lock(out, detected_mode, local_luma_strength, local_grid_size, local_max_gain, local_temporal_radius, local_reference_blend, local_detail_protection, local_lowpass_passes)
        out = self._apply_local_chroma_lock(out, detected_mode, local_chroma_strength, local_grid_size, local_chroma_max_shift, local_temporal_radius, local_reference_blend, local_chroma_detail_protection, local_lowpass_passes, local_chroma_threshold)
        out = self._apply_local_residual_pulse_lock(out, detected_mode, local_pulse_strength, local_grid_size, local_pulse_radius, local_pulse_luma_max_shift, local_pulse_chroma_max_shift, local_pulse_threshold, local_pulse_detail_protection, local_lowpass_passes, local_pulse_motion_protection)
        out = self._apply_static_background_lock(out, detected_mode, local_static_strength, local_grid_size, local_static_radius, local_static_luma_max_shift, local_static_chroma_max_shift, local_static_motion_threshold, local_lowpass_passes, local_static_neighbor_bias, local_static_detail_protection)
        seam_strength = max(0.0, min(1.0, float(seam_strength)))
        seam_frame_count = max(0, min(int(seam_frame_count), n))
        if detected_mode == "loop" and n >= 2 and seam_strength > 0.0 and seam_frame_count > 0:
            first_mean, first_std = self._stats(out[0]); last_mean, last_std = self._stats(out[-1])
            if bool(affect_end_frames):
                count = min(seam_frame_count, n)
                for j in range(count):
                    idx = n - count + j
                    ramp = ((j + 1) / float(count)) ** 1.35
                    corrected = self._match_frame(out[idx], first_mean, first_std, match_method)
                    out[idx] = self._blend_correction(out[idx], corrected, seam_strength * ramp, protect_highlights)
            if bool(affect_start_frames):
                count = min(seam_frame_count, n)
                for j in range(count):
                    ramp = (1.0 - (j / float(max(1, count)))) ** 1.35
                    corrected = self._match_frame(out[j], last_mean, last_std, match_method)
                    out[j] = self._blend_correction(out[j], corrected, seam_strength * ramp, protect_highlights)
        out = out.clamp(0.0, 1.0)
        if original.shape[-1] >= 4 and alpha is not None: result = torch.cat([out, alpha], dim=-1)
        elif original.shape[-1] == 1: result = self._luma(out).clamp(0.0, 1.0)
        else: result = out
        if single: result = result[:1]
        global_delta = float(sum(global_delta_vals) / len(global_delta_vals)) if global_delta_vals else 0.0
        return (result, str(detected_mode), float(seam_delta), global_delta)

class ShiroDownscaleFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "method": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact"], {"default": "lanczos"}),
                "snap_to_multiple_8": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "downscale"
    CATEGORY = "Shiro Tools/Video Management"

    def downscale(self, images, scale_factor=1.0, method="lanczos", snap_to_multiple_8=True):
        if images.dim() != 4: raise ValueError("IMAGE batch must be a 4D tensor [N,H,W,C].")
        original = images; b, h, w, c = original.shape
        sf = max(0.01, min(1.0, float(scale_factor)))
        multiple = 8 if bool(snap_to_multiple_8) else 1
        if sf >= 0.999999: return (original,)
        target_h = max(1, int(round(int(h) * sf))); target_w = max(1, int(round(int(w) * sf)))
        if multiple > 1:
            if target_h >= multiple: target_h = max(multiple, (target_h // multiple) * multiple)
            if target_w >= multiple: target_w = max(multiple, (target_w // multiple) * multiple)
        target_h = max(1, min(int(h), int(target_h))); target_w = max(1, min(int(w), int(target_w)))
        if target_h == int(h) and target_w == int(w): return (original,)
        x = original.permute(0, 3, 1, 2).contiguous(); method = str(method).lower()
        if method == "lanczos" or method == "bicubic": y = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        elif method == "bilinear": y = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
        elif method == "area": y = F.interpolate(x, size=(target_h, target_w), mode="area")
        elif method == "nearest-exact": y = F.interpolate(x, size=(target_h, target_w), mode="nearest-exact")
        else: y = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        return (y.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0),)

class ShiroWatermarkSimplified:
    """Versão simplificada do Watermark para Vídeo"""
    @classmethod
    def INPUT_TYPES(cls):
        files = _shiro_list_input_images()
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark": (files, {"image_upload": True}),
                "watermark_source": (["watermark_data", "selected_file"], {"default": "watermark_data"}),
                "position": (["top-left", "top-center", "top-right", "center-left", "center", "center-right", "bottom-left", "bottom-center", "bottom-right", "custom"], {"default": "bottom-right"}),
                "custom_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_mode": (["manual_width", "auto_short_side", "auto_long_side", "auto_area"], {"default": "manual_width"}),
                "scale": ("INT", {"default": 70, "min": 1, "max": 100, "step": 1}),
                "auto_scale_percent": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "min_size_px": ("INT", {"default": 96, "min": 1, "max": 16384, "step": 1}),
                "max_size_px": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "transparency": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 359, "step": 1}),
                "padding_x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "padding_y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "optical_padding": ("BOOLEAN", {"default": False}),
                "optical_strength": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "max_batch_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": { "watermark_data": ("SHIRO_WATERMARK",), }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Shiro Tools/Export"

    def apply(self, image, watermark, watermark_source="watermark_data", position="bottom-right", custom_x=1.0, custom_y=1.0, scale_mode="manual_width", scale=70, auto_scale_percent=10, min_size_px=96, max_size_px=512, transparency=100, rotation=0, padding_x=0, padding_y=0, optical_padding=False, optical_strength=80, max_batch_size=0, watermark_data=None):
        if not torch.is_tensor(image): raise ValueError("image must be a ComfyUI IMAGE tensor.")
        single = False
        if image.dim() == 3:
            image = image.unsqueeze(0); single = True
        if image.dim() != 4: raise ValueError(f"Expected IMAGE shape [B,H,W,C] or [H,W,C], got {tuple(image.shape)}.")
        b, h, w, c = image.shape
        if c not in (1, 3, 4): raise ValueError(f"Unsupported input channels: {c}. Expected 1, 3, or 4.")
        source = str(watermark_source or "watermark_data").lower().strip(); wm = None
        if source == "watermark_data":
            if watermark_data is None: raise ValueError("watermark_source is watermark_data, but watermark_data is not connected.")
            if isinstance(watermark_data, dict) and "pil" in watermark_data: wm = watermark_data["pil"].copy().convert("RGBA")
            elif isinstance(watermark_data, Image.Image): wm = watermark_data.copy().convert("RGBA")
            else: raise ValueError("watermark_data must come from Load Watermark.")
        if wm is None:
            if not watermark or not folder_paths.exists_annotated_filepath(watermark): raise ValueError(f"Invalid watermark image: {watermark}")
            path = folder_paths.get_annotated_filepath(watermark)
            try: wm = Image.open(path).convert("RGBA")
            except Exception as e: raise ValueError(f"Could not load watermark image '{watermark}': {e}")
        scale = max(1, min(100, int(scale))); auto_scale_percent = max(1, min(100, int(auto_scale_percent)))
        min_size_px = max(1, int(min_size_px)); max_size_px = max(min_size_px, int(max_size_px))
        transparency = max(0, min(100, int(transparency))); rotation = int(rotation) % 360
        padding_x = max(0, int(padding_x)); padding_y = max(0, int(padding_y)); max_batch_size = max(0, int(max_batch_size))
        target_w, target_h = _shiro_watermark_auto_target_width(w, h, wm.width, wm.height, scale_mode, scale, auto_scale_percent, min_size_px, max_size_px)
        wm = wm.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if transparency < 100:
            alpha = wm.getchannel("A")
            alpha = alpha.point(lambda px: int(px * transparency / 100.0))
            wm.putalpha(alpha)
        if rotation: wm = wm.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        pad_x, pad_y = _shiro_watermark_optical_padding(position, padding_x, padding_y, wm.width, wm.height, optical_padding, optical_strength)
        x, y = _shiro_watermark_position(position, w, h, wm.width, wm.height, pad_x, pad_y, custom_x, custom_y)
        out = []; output_channels = 3
        step = b if max_batch_size <= 0 else max(1, max_batch_size)
        for start in range(0, b, step):
            end = min(b, start + step)
            for i in range(start, end):
                base = _shiro_image_tensor_to_pil(image[i])
                layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
                layer.alpha_composite(wm, dest=(int(x), int(y)))
                composed = Image.alpha_composite(base, layer)
                out.append(_shiro_pil_to_image_tensor(composed, output_channels))
        result = torch.stack(out, dim=0).contiguous()
        if single: result = result[:1]
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ShiroAutoSeamLoopCut": ShiroAutoSeamLoopCut,
    "ShiroMotionDeStutterLoopCleaner": ShiroMotionDeStutterLoopCleaner,
    "ShiroBoundaryDeStutterLoopCleaner": ShiroBoundaryDeStutterLoopCleaner,
    "ShiroLastFirstRIFEContext": ShiroLastFirstRIFEContext,
    "ShiroAppendRIFEBridgeFrames": ShiroAppendRIFEBridgeFrames,
    "ShiroPreInterpolationLoopBoundaryCleaner": ShiroPreInterpolationLoopBoundaryCleaner,
    "ShiroColorBalance": ShiroColorBalance,
    "ShiroDownscaleFrames": ShiroDownscaleFrames,
    "ShiroWatermarkSimplified": ShiroWatermarkSimplified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroAutoSeamLoopCut": "Auto Seam Loop Cut",
    "ShiroMotionDeStutterLoopCleaner": "Motion De-Stutter Loop Cleaner",
    "ShiroBoundaryDeStutterLoopCleaner": "Boundary De-Stutter Cleaner",
    "ShiroLastFirstRIFEContext": "Last-First RIFE Context",
    "ShiroAppendRIFEBridgeFrames": "Apply RIFE Bridge Frames",
    "ShiroPreInterpolationLoopBoundaryCleaner": "Pre-Interpolation Boundary Cleaner",
    "ShiroColorBalance": "Color Balance",
    "ShiroDownscaleFrames": "Downscale Frames",
    "ShiroWatermarkSimplified": "Watermark (Simplified)",
}