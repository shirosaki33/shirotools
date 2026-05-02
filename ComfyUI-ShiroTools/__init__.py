
import time
import math
import torch
import torch.nn.functional as F


class ShiroImageDelay:
    """Pass an IMAGE through after waiting for the configured delay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "delay_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "enabled": ("BOOLEAN", {"default": True}),
                "always_run": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "delay"
    CATEGORY = "Shiro Tools/Timing"

    @classmethod
    def IS_CHANGED(cls, image, delay_seconds=5.0, enabled=True, always_run=True):
        if bool(always_run):
            return time.time()
        return f"{float(delay_seconds):.3f}:{bool(enabled)}"

    def delay(self, image, delay_seconds=5.0, enabled=True, always_run=True):
        if bool(enabled):
            seconds = max(0.0, float(delay_seconds))
            if seconds > 0.0:
                time.sleep(seconds)
        return (image,)


class ShiroPassthroughDelay:
    """Pass any ComfyUI data through after waiting for the configured delay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*",),
                "delay_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "enabled": ("BOOLEAN", {"default": True}),
                "always_run": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "delay"
    CATEGORY = "Shiro Tools/Timing"

    @classmethod
    def IS_CHANGED(cls, anything, delay_seconds=5.0, enabled=True, always_run=True):
        if bool(always_run):
            return time.time()
        return f"{float(delay_seconds):.3f}:{bool(enabled)}"

    def delay(self, anything, delay_seconds=5.0, enabled=True, always_run=True):
        if bool(enabled):
            seconds = max(0.0, float(delay_seconds))
            if seconds > 0.0:
                time.sleep(seconds)
        return (anything,)


def _clamp_int(value, low, high):
    return max(low, min(high, int(value)))


def _prepare_score_features(images, max_side=96, use_luma=True):
    """Return downscaled CPU tensor [N,C,H,W] and edge map [N,1,H,W]."""
    x = images.detach().float().cpu()
    if x.dim() != 4:
        raise ValueError("IMAGE batch must be a 4D tensor [N,H,W,C].")

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

    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    return x, edge


def _mse(a, b):
    return float(torch.mean((a - b) ** 2).item())


def _motion_score(feat, end_idx, start_idx, window):
    n = feat.shape[0]
    vals = []
    for k in range(1, int(window) + 1):
        a0 = end_idx - k
        a1 = end_idx - k + 1
        b0 = start_idx + k - 1
        b1 = start_idx + k
        if a0 < 0 or a1 < 0 or b0 >= n or b1 >= n:
            continue
        pre = feat[a1] - feat[a0]
        post = feat[b1] - feat[b0]
        vals.append(_mse(pre, post))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _build_candidates(n, start_from, start_to, end_from, end_to, min_loop_frames):
    if start_to <= 0:
        start_to = max(0, int(round(n * 0.35)))
    if end_from <= 0:
        end_from = max(0, int(round(n * 0.55)))
    if end_to <= 0:
        end_to = n - 1

    start_from = _clamp_int(start_from, 0, n - 1)
    start_to = _clamp_int(start_to, start_from, n - 1)
    end_from = _clamp_int(end_from, 0, n - 1)
    end_to = _clamp_int(end_to, end_from, n - 1)
    min_loop_frames = max(2, int(min_loop_frames))

    candidates = []
    for s in range(start_from, start_to + 1):
        min_e = max(end_from, s + min_loop_frames - 1)
        if min_e > end_to:
            continue
        for e in range(min_e, end_to + 1):
            candidates.append((s, e))
    return candidates


class ShiroAutoSeamLoopCut:
    """Find a less visible loop cut and return the cut video plus a short seam preview."""

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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("loop_frames", "seam_preview", "best_start", "best_end", "best_score")
    FUNCTION = "find_and_cut"
    CATEGORY = "Shiro Tools/Video Loop"

    def find_and_cut(
        self,
        images,
        start_search_from=0,
        start_search_to_0_auto=0,
        end_search_from_0_auto=0,
        end_search_to_0_auto=0,
        min_loop_frames=24,
        compare_window=4,
        preview_window=12,
        appearance_weight=1.0,
        motion_weight=2.0,
        edge_weight=0.5,
        max_eval_pairs=12000,
        use_luma=True,
    ):
        if not torch.is_tensor(images) or images.dim() != 4 or images.shape[0] < 2:
            return (images, images, 0, max(0, int(images.shape[0]) - 1), 0.0)

        feat, edge = _prepare_score_features(images, max_side=96, use_luma=use_luma)
        n = int(feat.shape[0])
        candidates = _build_candidates(
            n,
            start_search_from,
            start_search_to_0_auto,
            end_search_from_0_auto,
            end_search_to_0_auto,
            min_loop_frames,
        )
        if not candidates:
            return (images, images, 0, n - 1, 0.0)

        if len(candidates) > int(max_eval_pairs):
            step = max(1, int(math.ceil(len(candidates) / float(max_eval_pairs))))
            candidates = candidates[::step]

        best = None
        best_score = None
        for s, e in candidates:
            appearance = _mse(feat[e], feat[s])
            motion = _motion_score(feat, e, s, compare_window)
            edge_delta = _mse(edge[e], edge[s])
            score = appearance_weight * appearance + motion_weight * motion + edge_weight * edge_delta
            if best_score is None or score < best_score:
                best_score = score
                best = (s, e)

        if best is None:
            return (images, images, 0, n - 1, 0.0)

        s, e = best
        loop_frames = images[s:e + 1]
        pw = max(1, int(preview_window))
        end_preview = images[max(s, e - pw + 1):e + 1]
        start_preview = images[s:min(e + 1, s + pw)]
        seam_preview = torch.cat([end_preview, start_preview], dim=0)
        return (loop_frames, seam_preview, int(s), int(e), float(best_score))


class ShiroMotionDeStutterLoopCleaner:
    """Drop a few very-low-motion frames to reduce micro-stutter before loop playback."""

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
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "dropped_count", "motion_cutoff", "median_motion")
    FUNCTION = "clean"
    CATEGORY = "Shiro Tools/Video Loop"

    def clean(
        self,
        images,
        low_motion_ratio=0.35,
        protected_start_frames=2,
        protected_end_frames=2,
        max_frames_to_drop=3,
        max_consecutive_drops=2,
        enabled=True,
    ):
        if (not bool(enabled)) or (not torch.is_tensor(images)) or images.dim() != 4:
            return (images, 0, 0.0, 0.0)

        n = int(images.shape[0])
        if n < 3:
            return (images, 0, 0.0, 0.0)

        x = images.detach().float().cpu()
        # mean absolute frame-to-frame difference; score i refers to frame i vs i-1
        diffs = torch.mean(torch.abs(x[1:] - x[:-1]), dim=(1, 2, 3))
        scores = [0.0] + [float(v.item()) for v in diffs]

        protected_start_frames = _clamp_int(protected_start_frames, 0, max(0, n - 1))
        protected_end_frames = _clamp_int(protected_end_frames, 0, max(0, n - 1))
        max_frames_to_drop = max(0, int(max_frames_to_drop))
        max_consecutive_drops = max(1, int(max_consecutive_drops))

        valid_start = max(1, protected_start_frames)
        valid_end = max(valid_start, n - protected_end_frames)
        valid_indices = list(range(valid_start, valid_end))
        if not valid_indices or max_frames_to_drop == 0:
            median_motion = float(torch.tensor(scores[1:]).median().item()) if n > 1 else 0.0
            return (images, 0, 0.0, median_motion)

        valid_vals = [scores[i] for i in valid_indices]
        median_motion = float(torch.tensor(valid_vals, dtype=torch.float32).median().item()) if valid_vals else 0.0
        motion_cutoff = float(max(0.0, median_motion * float(low_motion_ratio)))

        candidates = [i for i in valid_indices if scores[i] <= motion_cutoff]
        candidates.sort(key=lambda idx: (scores[idx], idx))

        selected = []
        selected_set = set()
        for idx in candidates:
            if len(selected) >= max_frames_to_drop:
                break

            # avoid removing too many adjacent frames in one cluster
            run = 1
            left = idx - 1
            while left in selected_set:
                run += 1
                left -= 1
            right = idx + 1
            while right in selected_set:
                run += 1
                right += 1
            if run > max_consecutive_drops:
                continue

            selected.append(idx)
            selected_set.add(idx)

        if not selected:
            return (images, 0, motion_cutoff, median_motion)

        keep = [i for i in range(n) if i not in selected_set]
        if len(keep) < 2:
            return (images, 0, motion_cutoff, median_motion)

        cleaned = images[keep]
        return (cleaned, len(selected), motion_cutoff, median_motion)


class ShiroBoundaryDeStutterLoopCleaner:
    """Trim low-motion duplicate-like frames from the loop boundaries only."""

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
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "start_dropped", "end_dropped", "motion_cutoff", "median_motion")
    FUNCTION = "clean"
    CATEGORY = "Shiro Tools/Video Loop"

    def clean(
        self,
        images,
        low_motion_ratio=0.60,
        start_scan_frames=12,
        end_scan_frames=12,
        max_start_drop=5,
        max_end_drop=3,
        min_remaining_frames=16,
        enabled=True,
    ):
        if (not bool(enabled)) or (not torch.is_tensor(images)) or images.dim() != 4:
            return (images, 0, 0, 0.0, 0.0)

        n = int(images.shape[0])
        if n < 3:
            return (images, 0, 0, 0.0, 0.0)

        x = images.detach().float().cpu()
        diffs = torch.mean(torch.abs(x[1:] - x[:-1]), dim=(1, 2, 3))
        vals = [float(v.item()) for v in diffs]
        if not vals:
            return (images, 0, 0, 0.0, 0.0)

        median_motion = float(torch.tensor(vals, dtype=torch.float32).median().item())
        motion_cutoff = float(max(0.0, median_motion * float(low_motion_ratio)))

        start_scan_frames = max(1, int(start_scan_frames))
        end_scan_frames = max(1, int(end_scan_frames))
        max_start_drop = max(0, int(max_start_drop))
        max_end_drop = max(0, int(max_end_drop))
        min_remaining_frames = max(2, int(min_remaining_frames))

        # Start: if 0->1, 1->2, etc. are very low motion, drop the earlier frame.
        start_drop = 0
        for i in range(0, min(start_scan_frames, len(vals))):
            if start_drop >= max_start_drop:
                break
            if n - (start_drop + 1) < min_remaining_frames:
                break
            if vals[i] <= motion_cutoff:
                start_drop += 1
            else:
                break

        # End: if near-final transitions are very low motion, drop final frames.
        end_drop = 0
        start_limit = max(0, len(vals) - end_scan_frames)
        for i in range(len(vals) - 1, start_limit - 1, -1):
            if end_drop >= max_end_drop:
                break
            if n - start_drop - (end_drop + 1) < min_remaining_frames:
                break
            if vals[i] <= motion_cutoff:
                end_drop += 1
            else:
                break

        if start_drop == 0 and end_drop == 0:
            return (images, 0, 0, motion_cutoff, median_motion)

        end_index = n - end_drop
        cleaned = images[start_drop:end_index].contiguous()
        return (cleaned, int(start_drop), int(end_drop), motion_cutoff, median_motion)





class ShiroLastFirstContextBridge:
    """Advanced no-model bridge between the final frame and the first frame using local temporal context."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_before": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "context_after": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "bridge_frame_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "method": (["balanced_anti_first", "balanced", "motion_extrapolated", "catmull_rom", "bezier", "linear"], {"default": "balanced_anti_first"}),
                "max_first_progress": ("FLOAT", {"default": 0.72, "min": 0.05, "max": 0.98, "step": 0.01}),
                "anti_first_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "insert_mode": (["append", "replace_last", "replace_last_and_append"], {"default": "append"}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "bridge_frames")
    FUNCTION = "append_bridge"
    CATEGORY = "Shiro Tools/Video Loop"

    def _catmull_rom(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        out = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        return out.clamp(0.0, 1.0)

    def _bezier(self, last, first, vel_pre, vel_post, t):
        # Cubic Bezier from last to first.
        # Control points follow the incoming motion at the end and the outgoing motion at the start.
        c1 = (last + vel_pre * 0.50).clamp(0.0, 1.0)
        c2 = (first - vel_post * 0.50).clamp(0.0, 1.0)
        u = 1.0 - t
        return (u**3 * last + 3.0 * u*u*t * c1 + 3.0 * u*t*t * c2 + t**3 * first).clamp(0.0, 1.0)

    def _motion_extrapolated(self, last, first, vel_pre, vel_post, t):
        # Estimate the bridge from both directions:
        #   forward: last + incoming end velocity
        #   backward: first - outgoing start velocity
        # Then blend both predictions according to t.
        forward = (last + vel_pre * t).clamp(0.0, 1.0)
        backward = (first - vel_post * (1.0 - t)).clamp(0.0, 1.0)
        linear = (last * (1.0 - t) + first * t).clamp(0.0, 1.0)
        out = (forward * (1.0 - t) + backward * t) * 0.70 + linear * 0.30
        return out.clamp(0.0, 1.0)

    def append_bridge(self, images, context_before=3, context_after=3, bridge_frame_count=1, method="balanced_anti_first", max_first_progress=0.72, anti_first_strength=0.65, insert_mode="append", enabled=True):
        if (not bool(enabled)) or (not torch.is_tensor(images)) or images.dim() != 4:
            return (images, images[:0])

        n = int(images.shape[0])
        count = max(0, int(bridge_frame_count))
        if count <= 0 or n < 2:
            return (images.contiguous(), images[:0].contiguous())

        before = max(1, min(int(context_before), max(1, n - 1)))
        after = max(1, min(int(context_after), max(1, n - 1)))

        first = images[0].float()
        last = images[-1].float()
        prev = images[-2].float() if n >= 2 else last
        nxt = images[1].float() if n >= 2 else first

        # Average velocity near the loop boundary.
        # This makes the bridge less dependent on a single possibly-noisy neighboring frame.
        pre_ref = images[max(0, n - 1 - before)].float()
        post_ref = images[min(n - 1, after)].float()
        vel_pre = (last - pre_ref) / float(before)
        vel_post = (post_ref - first) / float(after)

        bridges = []
        for i in range(1, count + 1):
            t = float(i) / float(count + 1)
            linear = (last * (1.0 - t) + first * t).clamp(0.0, 1.0)
            cat = self._catmull_rom(prev, last, first, nxt, t)
            mot = self._motion_extrapolated(last, first, vel_pre, vel_post, t)
            bez = self._bezier(last, first, vel_pre, vel_post, t)

            if method == "linear":
                b = linear
            elif method == "catmull_rom":
                b = cat
            elif method == "motion_extrapolated":
                b = mot
            elif method == "bezier":
                b = bez
            else:  # balanced / balanced_anti_first
                # Stable base: combines curve continuity, motion estimate, and a little direct blend.
                b = (cat * 0.40 + mot * 0.40 + linear * 0.20).clamp(0.0, 1.0)

                if method == "balanced_anti_first":
                    # Guard against a bridge that visually becomes almost the same as frame 0.
                    # progress ~= 0 means closer to last; progress ~= 1 means closer to first.
                    d_last = torch.mean((b - last) ** 2)
                    d_first = torch.mean((b - first) ** 2)
                    denom = d_last + d_first
                    if float(denom.item()) > 1e-12:
                        progress = float((d_last / denom).item())
                        max_p = max(0.05, min(0.98, float(max_first_progress)))
                        strength = max(0.0, min(1.0, float(anti_first_strength)))
                        if progress > max_p and strength > 0.0:
                            t_guard = max(0.05, min(t, t * 0.55))
                            linear_guard = (last * (1.0 - t_guard) + first * t_guard).clamp(0.0, 1.0)
                            mot_guard = self._motion_extrapolated(last, first, vel_pre, vel_post, t_guard)
                            guard = (mot_guard * 0.65 + linear_guard * 0.35).clamp(0.0, 1.0)
                            b = (b * (1.0 - strength) + guard * strength).clamp(0.0, 1.0)

            bridges.append(b.unsqueeze(0).to(dtype=images.dtype, device=images.device))

        bridge_batch = torch.cat(bridges, dim=0).contiguous() if bridges else images[:0].contiguous()

        if insert_mode == "append":
            out = torch.cat((images, bridge_batch), dim=0).contiguous()
        elif insert_mode == "replace_last":
            out = torch.cat((images[:-1], bridge_batch[-1:].contiguous()), dim=0).contiguous()
        else:  # replace_last_and_append
            out = torch.cat((images[:-1], bridge_batch), dim=0).contiguous()

        return (out, bridge_batch)



class ShiroLastFirstRIFEContext:
    """Build a tiny boundary context batch around last->first for local RIFE bridge generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_before": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "context_after": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "bridge_frame_count": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "rife_multiplier": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("original_images", "rife_context_frames", "rife_multiplier", "bridge_frame_count", "boundary_segment_index")
    FUNCTION = "build_context"
    CATEGORY = "Shiro Tools/Video Loop"

    def build_context(self, images, context_before=3, context_after=3, bridge_frame_count=1, rife_multiplier=4, enabled=True):
        c = max(1, int(bridge_frame_count))
        m = max(c + 1, int(rife_multiplier))
        before = max(1, int(context_before))
        after = max(1, int(context_after))
        boundary_segment_index = before - 1

        if (not bool(enabled)) or (not torch.is_tensor(images)) or images.dim() != 4:
            return (images, images, int(m), int(c), int(boundary_segment_index))

        n = int(images.shape[0])
        if n <= 0:
            return (images, images, int(m), int(c), int(boundary_segment_index))

        frames = []

        # Context before the seam: [-before, ..., -1]
        # Clamp/wrap behavior keeps the node usable even on very short videos.
        for k in range(before, 0, -1):
            idx = max(0, n - k)
            frames.append(images[idx])

        # Context after the seam: [0, 1, ..., after]
        # Includes frame 0 because the target RIFE segment is last -> first.
        for k in range(0, after + 1):
            idx = min(n - 1, k)
            frames.append(images[idx])

        context = torch.stack(frames, dim=0).contiguous()
        return (images.contiguous(), context, int(m), int(c), int(boundary_segment_index))


class ShiroAppendRIFEBridgeFrames:
    """Extract local RIFE frames between last and first, then insert them into the final loop."""

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
                "insert_mode": (["append", "replace_last", "replace_last_and_append"], {"default": "replace_last_and_append"}),
                "min_distance_from_first": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0005}),
                "max_first_progress": ("FLOAT", {"default": 0.72, "min": 0.05, "max": 0.98, "step": 0.01}),
                "first_penalty_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "bridge_frames")
    FUNCTION = "append"
    CATEGORY = "Shiro Tools/Video Loop"

    def _mse_img(self, a, b):
        return float(torch.mean((a.float() - b.float()) ** 2).item())

    def _pick_candidates(self, candidates, original_images, count, pick_mode, min_distance_from_first=0.001, max_first_progress=0.72, first_penalty_strength=2.0):
        total = int(candidates.shape[0])
        count = max(1, min(int(count), total))

        if total <= count:
            return candidates[:count]

        if pick_mode == "first":
            return candidates[:count]
        if pick_mode == "last":
            return candidates[-count:]
        if pick_mode == "middle":
            center = total // 2
            start = max(0, min(total - count, center - (count // 2)))
            return candidates[start:start + count]

        # progress ~= 0 means closer to last; progress ~= 1 means closer to first.
        # Anti-first modes add a penalty when a RIFE candidate becomes too similar to frame 0.
        last = original_images[-1]
        first = original_images[0]
        progress = []
        d_first_values = []
        for i in range(total):
            f = candidates[i]
            d_last = self._mse_img(f, last)
            d_first = self._mse_img(f, first)
            denom = d_last + d_first
            if denom <= 1e-12:
                p = (i + 1) / float(total + 1)
            else:
                p = d_last / denom
            progress.append(float(p))
            d_first_values.append(float(d_first))

        max_p = max(0.05, min(0.98, float(max_first_progress)))
        min_d_first = max(0.0, float(min_distance_from_first))
        penalty_strength = max(0.0, float(first_penalty_strength))

        selected = []
        used = set()
        for j in range(1, count + 1):
            base_target = j / float(count + 1)
            if pick_mode == "prefer_last_side":
                target = base_target * 0.70
            elif pick_mode == "anti_first_bias":
                target = min(base_target * 0.85, max_p)
            else:
                target = base_target

            best_i = None
            best_score = None
            for i, p in enumerate(progress):
                if i in used:
                    continue
                score = abs(p - target)
                if pick_mode in ("anti_first_bias", "prefer_last_side"):
                    if p > max_p:
                        score += (p - max_p) * penalty_strength
                    if min_d_first > 0.0 and d_first_values[i] < min_d_first:
                        score += ((min_d_first - d_first_values[i]) / min_d_first) * penalty_strength
                if best_score is None or score < best_score:
                    best_score = score
                    best_i = i
            if best_i is not None:
                selected.append(best_i)
                used.add(best_i)

        if not selected:
            center = total // 2
            selected = [center]
        selected = sorted(selected)
        return candidates[selected]

    def append(
        self,
        original_images,
        rife_context_result,
        bridge_frame_count=1,
        rife_multiplier=4,
        boundary_segment_index=2,
        pick_mode="anti_first_bias",
        insert_mode="replace_last_and_append",
        min_distance_from_first=0.001,
        max_first_progress=0.72,
        first_penalty_strength=2.0,
        enabled=True,
    ):
        if (not bool(enabled)) or (not torch.is_tensor(original_images)) or original_images.dim() != 4:
            empty = original_images[:0] if torch.is_tensor(original_images) and original_images.dim() == 4 else original_images
            return (original_images, empty)

        count = max(1, int(bridge_frame_count))
        m = max(count + 1, int(rife_multiplier))
        seg = max(0, int(boundary_segment_index))

        if not torch.is_tensor(rife_context_result) or rife_context_result.dim() != 4:
            return (original_images.contiguous(), original_images[:0].contiguous())

        n = int(rife_context_result.shape[0])

        # Expected FL_RIFE layout for context frames and multiplier m:
        # c0, generated..., c1, generated..., c2, ...
        # If last->first is segment `seg`, generated candidates are:
        # seg*m + 1 ... (seg+1)*m - 1
        start = seg * m + 1
        end = (seg + 1) * m

        if n >= end and start < end:
            candidates = rife_context_result[start:end]
        else:
            # Fallback for RIFE variants with different layout: use central candidates.
            center = n // 2
            window = max(count, m - 1)
            s = max(0, center - window // 2)
            e = min(n, s + window)
            s = max(0, e - window)
            candidates = rife_context_result[s:e]

        if candidates.shape[0] <= 0:
            return (original_images.contiguous(), original_images[:0].contiguous())

        bridge = self._pick_candidates(
            candidates,
            original_images,
            count,
            pick_mode,
            min_distance_from_first=min_distance_from_first,
            max_first_progress=max_first_progress,
            first_penalty_strength=first_penalty_strength,
        ).to(dtype=original_images.dtype).contiguous()

        if insert_mode == "append":
            out = torch.cat((original_images, bridge), dim=0).contiguous()
        elif insert_mode == "replace_last":
            out = torch.cat((original_images[:-1], bridge[:1]), dim=0).contiguous()
            bridge = bridge[:1].contiguous()
        else:  # replace_last_and_append
            out = torch.cat((original_images[:-1], bridge), dim=0).contiguous()

        return (out, bridge)


NODE_CLASS_MAPPINGS = {
    "ShiroImageDelay": ShiroImageDelay,
    "ShiroPassthroughDelay": ShiroPassthroughDelay,
    "ShiroAutoSeamLoopCut": ShiroAutoSeamLoopCut,
    "ShiroMotionDeStutterLoopCleaner": ShiroMotionDeStutterLoopCleaner,
    "ShiroBoundaryDeStutterLoopCleaner": ShiroBoundaryDeStutterLoopCleaner,
    "ShiroLastFirstContextBridge": ShiroLastFirstContextBridge,
    "ShiroLastFirstRIFEContext": ShiroLastFirstRIFEContext,
    "ShiroAppendRIFEBridgeFrames": ShiroAppendRIFEBridgeFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroImageDelay": "Image Workflow Delay (Shiro)",
    "ShiroPassthroughDelay": "Passthrough Workflow Delay (Shiro)",
    "ShiroAutoSeamLoopCut": "Auto Seam Loop Cut (Shiro)",
    "ShiroMotionDeStutterLoopCleaner": "Motion De-Stutter Loop Cleaner (Shiro)",
    "ShiroBoundaryDeStutterLoopCleaner": "Boundary De-Stutter Loop Cleaner (Shiro)",
    "ShiroLastFirstContextBridge": "Last-First Context Bridge Advanced (Shiro)",
    "ShiroLastFirstRIFEContext": "Last-First RIFE Context (Shiro)",
    "ShiroAppendRIFEBridgeFrames": "Append RIFE Bridge Frames (Shiro)",
}
