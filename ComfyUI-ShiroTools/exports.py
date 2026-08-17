import os
import re
import json
import torch
import numpy as np
from PIL import Image
import folder_paths

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (Export e Watermark)
# ---------------------------------------------------------------------------

def _shiro_list_input_images():
    input_dir = folder_paths.get_input_directory()
    try:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))
    except Exception:
        return []

def _shiro_image_tensor_to_pil(frame):
    x = frame.detach().float().cpu().clamp(0, 1)
    arr = (x.numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim != 3: raise ValueError("Expected one image frame with shape [H,W,C].")
    c = arr.shape[-1]
    if c == 1: return Image.fromarray(arr[:, :, 0], mode="L").convert("RGBA")
    if c == 3: return Image.fromarray(arr, mode="RGB").convert("RGBA")
    if c == 4: return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"Unsupported image channel count: {c}. Expected 1, 3, or 4.")

def _shiro_pil_to_image_tensor(img, channels=3):
    if channels == 1: arr = np.asarray(img.convert("L"), dtype=np.float32)[:, :, None] / 255.0
    elif channels == 4: arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    else: arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).float()

def _shiro_watermark_position(position, base_w, base_h, wm_w, wm_h, padding_x, padding_y, custom_x=1.0, custom_y=1.0):
    pos = str(position or "bottom-right").lower().strip().replace("_", "-")
    aliases = {"middle": "center", "centre": "center", "top-middle": "top-center", "bottom-middle": "bottom-center",
               "middle-left": "center-left", "middle-right": "center-right", "left-center": "center-left",
               "right-center": "center-right", "center-top": "top-center", "center-bottom": "bottom-center"}
    pos = aliases.get(pos, pos)
    if pos == "custom":
        cx = max(0.0, min(1.0, float(custom_x))); cy = max(0.0, min(1.0, float(custom_y)))
        max_x = max(0, int(base_w) - int(wm_w)); max_y = max(0, int(base_h) - int(wm_h))
        return int(round(max_x * cx)), int(round(max_y * cy))
    x_positions = {"top-left": int(padding_x), "center-left": int(padding_x), "bottom-left": int(padding_x),
                   "top-center": (base_w - wm_w) // 2, "center": (base_w - wm_w) // 2, "bottom-center": (base_w - wm_w) // 2,
                   "top-right": base_w - wm_w - int(padding_x), "center-right": base_w - wm_w - int(padding_x),
                   "bottom-right": base_w - wm_w - int(padding_x)}
    y_positions = {"top-left": int(padding_y), "top-center": int(padding_y), "top-right": int(padding_y),
                   "center-left": (base_h - wm_h) // 2, "center": (base_h - wm_h) // 2, "center-right": (base_h - wm_h) // 2,
                   "bottom-left": base_h - wm_h - int(padding_y), "bottom-center": base_h - wm_h - int(padding_y),
                   "bottom-right": base_h - wm_h - int(padding_y)}
    if pos not in x_positions or pos not in y_positions: pos = "bottom-right"
    return x_positions[pos], y_positions[pos]

def _shiro_watermark_target_size(base_w, base_h, wm_w, wm_h, scale_percent):
    scale_percent = max(0, min(100, float(scale_percent)))
    short_side = max(1, min(int(base_w), int(base_h)))
    target_short = max(1, int(round(short_side * (scale_percent / 100.0))))
    if wm_w <= wm_h:
        target_w = target_short
        target_h = max(1, int(round(wm_h * target_w / max(1, wm_w))))
    else:
        target_h = target_short
        target_w = max(1, int(round(wm_w * target_h / max(1, wm_h))))
    if target_w > base_w:
        target_w = int(base_w); target_h = max(1, int(round(wm_h * target_w / max(1, wm_w))))
    if target_h > base_h:
        target_h = int(base_h); target_w = max(1, int(round(wm_w * target_h / max(1, wm_h))))
    return int(target_w), int(target_h)

def _shiro_watermark_optical_padding(position, padding_x, padding_y, wm_w, wm_h, optical_padding, optical_strength):
    if not bool(optical_padding): return int(padding_x), int(padding_y)
    strength = max(0, min(100, int(optical_strength))) / 100.0
    pos = str(position or "bottom-right").lower().strip().replace("_", "-")
    if pos in ("center", "middle", "centre"): return int(padding_x), int(padding_y)
    offset_x = min(max(0, int(padding_x)) + int(round(wm_w * 0.06 * strength)), max(0, int(padding_x)) + 96)
    offset_y = min(max(0, int(padding_y)) + int(round(wm_h * 0.06 * strength)), max(0, int(padding_y)) + 96)
    return offset_x, offset_y

import torch.nn.functional as F

def _shiro_watermark_activity_map(image, sample_max=4):
    b = int(image.shape[0])
    if b <= sample_max: idxs = list(range(b))
    else: idxs = sorted(set(int(round(v)) for v in np.linspace(0, b - 1, sample_max)))
    frames = image[idxs].detach().float().cpu()
    if frames.shape[-1] >= 3:
        w = torch.tensor([0.299, 0.587, 0.114], dtype=frames.dtype).view(1, 1, 1, 3)
        luma = (frames[..., :3] * w).sum(dim=-1)
    else: luma = frames[..., 0]
    dx = torch.abs(luma[:, :, 1:] - luma[:, :, :-1]); dx = F.pad(dx, (0, 1, 0, 0))
    dy = torch.abs(luma[:, 1:, :] - luma[:, :-1, :]); dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    if luma.shape[0] > 1: temporal = luma.std(dim=0)
    else: temporal = torch.zeros_like(luma[0])
    return edge.mean(dim=0) + temporal

def _shiro_watermark_region_score(activity, x, y, w, h):
    H, Wd = activity.shape
    x0 = max(0, min(int(x), Wd - 1)); y0 = max(0, min(int(y), H - 1))
    x1 = max(x0 + 1, min(int(x + w), Wd)); y1 = max(y0 + 1, min(int(y + h), H))
    region = activity[y0:y1, x0:x1]
    if region.numel() == 0: return float("inf")
    return float(region.mean().item())

_AUTO_CANDIDATES = {
    "auto_full": ["bottom-right", "bottom-left", "top-right", "top-left", "bottom-center", "top-center", "center-right", "center-left", "center"],
    "auto_corner": ["bottom-right", "bottom-left", "top-right", "top-left"],
    "auto_middle": ["bottom-center", "top-center", "center-right", "center-left", "center"],
}

def _shiro_watermark_auto_position(image, target_w, target_h, padding_x, padding_y, mode="auto_full"):
    candidates = _AUTO_CANDIDATES.get(mode, _AUTO_CANDIDATES["auto_full"])
    activity = _shiro_watermark_activity_map(image)
    H, Wd = activity.shape
    best_pos = candidates[0]; best_score = None
    for idx, pos in enumerate(candidates):
        x, y = _shiro_watermark_position(pos, Wd, H, target_w, target_h, padding_x, padding_y)
        score = _shiro_watermark_region_score(activity, x, y, target_w, target_h) + idx * 1e-6
        if best_score is None or score < best_score:
            best_score = score; best_pos = pos
    return best_pos

def _shiro_calculate_next_counter(folder_path):
    """Contador compartilhado por ShiroExportGlobalConfig e SaveExportA1111
    (helper unico, antes duplicado em cada classe)."""
    highest_number = 0
    if not os.path.exists(folder_path): return 1
    number_pattern = re.compile(r'(\d+)')
    for filename in os.listdir(folder_path):
        found_numbers = number_pattern.findall(filename)
        if found_numbers:
            for num_str in found_numbers:
                try:
                    value = int(num_str)
                    if value > highest_number: highest_number = value
                except ValueError: pass
    return highest_number + 1

# Cor de fundo usada quando a transparencia precisa virar opaca (JPG, ou
# keep_alpha desligado). Vindo do ShiroSaveExportLinkedTest / testes.
_SHIRO_ALPHA_FALLBACK_RGB = (114, 114, 114)

def _shiro_mask_to_alpha_image(img_rgb, mask_tensor):
    """Combina uma imagem RGB (PIL) com um frame de MASK do ComfyUI (tensor 2D,
    valores 0..1) pra formar uma imagem RGBA, seguindo a convencao do Load
    Image padrao do ComfyUI: mask = 1 - alpha (1 = area transparente), entao
    invertemos pra virar alpha de verdade. Redimensiona o mask se o tamanho
    nao bater com o da imagem (ex: upscale aplicado so na IMAGE, deixando o
    MASK no tamanho original)."""
    m = mask_tensor.detach().float().cpu().clamp(0, 1)
    mask_arr = (m.numpy() * 255.0).round().astype(np.uint8)
    mask_pil = Image.fromarray(mask_arr, mode="L")
    if mask_pil.size != img_rgb.size:
        mask_pil = mask_pil.resize(img_rgb.size, Image.Resampling.LANCZOS)
    mask_pil = Image.eval(mask_pil, lambda px: 255 - px)  # mask=1-alpha -> vira alpha
    out = img_rgb.convert("RGBA")
    out.putalpha(mask_pil)
    return out

def _shiro_save_frame_alpha(image_tensor, mask_tensor, file_path, format, quality, keep_alpha,
                             save_png_metadata=False, prompt=None, extra_pnginfo=None):
    """Salva um unico frame com suporte opcional a transparencia. A IMAGE do
    ComfyUI quase sempre vem em RGB (3 canais), com o alpha real separado no
    MASK. Se um mask for passado e a imagem ainda nao tiver alpha embutido,
    reconstruimos o canal alpha a partir dele antes de decidir o que fazer
    em cada formato."""
    arr = np.clip(255.0 * image_tensor.detach().float().cpu().numpy(), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    if mask_tensor is not None and img.mode != "RGBA":
        img = _shiro_mask_to_alpha_image(img.convert("RGB"), mask_tensor)

    has_alpha = img.mode == "RGBA"
    keep_alpha_here = has_alpha and bool(keep_alpha) and format in ("PNG", "WEBP")

    if has_alpha and not keep_alpha_here:
        # Formato sem suporte a alpha (JPG) ou keep_alpha desligado: compoe
        # sobre o fundo solido em vez de so descartar o canal alpha.
        bg = Image.new("RGB", img.size, _SHIRO_ALPHA_FALLBACK_RGB)
        bg.paste(img, mask=img.getchannel("A"))
        img = bg

    if format == "PNG":
        png_compress = max(0, min(9, round(9 - ((quality / 100.0) * 9))))
        if save_png_metadata:
            from PIL.PngImagePlugin import PngInfo
            metadata = PngInfo()
            if prompt is not None: metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo: metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            img.save(file_path, pnginfo=metadata, compress_level=png_compress)
        else:
            img.save(file_path, compress_level=png_compress)
    elif format in ("JPG", "JPEG"):
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.save(file_path, quality=quality, optimize=True)
    elif format == "WEBP":
        img.save(file_path, quality=quality, method=6)

# ---------------------------------------------------------------------------
# NODES (Export e Watermark)
# ---------------------------------------------------------------------------

class ShiroLoadWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        files = _shiro_list_input_images()
        return {"required": {"watermark": (files, {"image_upload": True})}}
    RETURN_TYPES = ("SHIRO_WATERMARK",)
    RETURN_NAMES = ("watermark_data",)
    FUNCTION = "load"
    CATEGORY = "Shiro Tools/Export"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs): return float("nan")

    def load(self, watermark):
        if not watermark or not folder_paths.exists_annotated_filepath(watermark): raise ValueError(f"Invalid watermark image: {watermark}")
        path = folder_paths.get_annotated_filepath(watermark)
        try: pil = Image.open(path).convert("RGBA")
        except Exception as e: raise ValueError(f"Could not load watermark image '{watermark}': {e}")
        return ({"pil": pil.copy(), "filename": watermark},)

class ShiroWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        files = _shiro_list_input_images()
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark": (files, {"image_upload": True}),
                "watermark_source": (["watermark_data", "selected_file"], {"default": "watermark_data"}),
                "position": (["auto_full", "auto_corner", "auto_middle",
                              "top-left", "top-center", "top-right",
                              "center-left", "center", "center-right",
                              "bottom-left", "bottom-center", "bottom-right", "custom"],
                             {"default": "auto_corner"}),
                "custom_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_percent": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1,
                                          "display_name": "Scale (% of the photo's shorter side)"}),
                "transparency": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 359, "step": 1}),
                "padding_x": ("INT", {"default": 24, "min": 0, "max": 16384, "step": 1}),
                "padding_y": ("INT", {"default": 24, "min": 0, "max": 16384, "step": 1}),
                "optical_padding": ("BOOLEAN", {"default": False}),
                "optical_strength": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "max_batch_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "watermark_data": ("SHIRO_WATERMARK",),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "chosen_position")
    FUNCTION = "apply"
    CATEGORY = "Shiro Tools/Export"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs): return float("nan")

    def apply(self, image, watermark, watermark_source="watermark_data", position="auto_corner", custom_x=1.0, custom_y=1.0,
              scale_percent=15, transparency=100, rotation=0, padding_x=24, padding_y=24,
              optical_padding=False, optical_strength=80, max_batch_size=0, watermark_data=None):
        if not torch.is_tensor(image): raise ValueError("image must be a ComfyUI IMAGE tensor.")
        single = False
        if image.dim() == 3:
            image = image.unsqueeze(0); single = True
        if image.dim() != 4: raise ValueError(f"Expected IMAGE shape [B,H,W,C] or [H,W,C], got {tuple(image.shape)}.")
        b, h, w, c = image.shape
        if c not in (1, 3, 4): raise ValueError(f"Unsupported input channels: {c}.")

        source = str(watermark_source or "watermark_data").lower().strip(); wm = None
        if source == "watermark_data":
            if watermark_data is None: raise ValueError("watermark_data is not connected.")
            if isinstance(watermark_data, dict) and "pil" in watermark_data: wm = watermark_data["pil"].copy().convert("RGBA")
            elif isinstance(watermark_data, Image.Image): wm = watermark_data.copy().convert("RGBA")
            else: raise ValueError("watermark_data must come from Load Watermark.")
        if wm is None:
            if not watermark or not folder_paths.exists_annotated_filepath(watermark): raise ValueError(f"Invalid watermark image: {watermark}")
            path = folder_paths.get_annotated_filepath(watermark)
            try: wm = Image.open(path).convert("RGBA")
            except Exception as e: raise ValueError(f"Could not load watermark: {e}")

        transparency = max(0, min(100, int(transparency))); rotation = int(rotation) % 360
        padding_x = max(0, int(padding_x)); padding_y = max(0, int(padding_y)); max_batch_size = max(0, int(max_batch_size))

        target_w, target_h = _shiro_watermark_target_size(w, h, wm.width, wm.height, scale_percent)
        wm = wm.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if transparency < 100:
            alpha = wm.getchannel("A")
            alpha = alpha.point(lambda px: int(px * transparency / 100.0))
            wm.putalpha(alpha)
        if rotation: wm = wm.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

        pos_lower = str(position or "auto_corner").lower().strip()
        if pos_lower in _AUTO_CANDIDATES:
            chosen_position = _shiro_watermark_auto_position(image, wm.width, wm.height, padding_x, padding_y, mode=pos_lower)
        else: chosen_position = position

        pad_x, pad_y = _shiro_watermark_optical_padding(chosen_position, padding_x, padding_y, wm.width, wm.height, optical_padding, optical_strength)
        x, y = _shiro_watermark_position(chosen_position, w, h, wm.width, wm.height, pad_x, pad_y, custom_x, custom_y)

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
        return (result, str(chosen_position))

class ShiroExportGlobalConfig:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_folder": ("STRING", {"default": "extra-imagem"}),
                "prefix": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("SHIRO_EXPORT_CONFIG",)
    RETURN_NAMES = ("export_config",)
    FUNCTION = "setup"
    CATEGORY = "Shiro Tools/Export"

    def setup(self, output_folder, prefix):
        from datetime import datetime
        hoje = datetime.now().strftime("%Y-%m-%d")
        output_folder = output_folder.replace("%date:yyyy-MM-dd%", hoje).replace("%date%", hoje)
        full_path = os.path.join(self.output_dir, output_folder)
        os.makedirs(full_path, exist_ok=True)
        counter = _shiro_calculate_next_counter(full_path)
        return ({"full_path": full_path, "prefix": prefix, "counter": counter},)

class ShiroSaveExportLinked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "export_config": ("SHIRO_EXPORT_CONFIG", ),
                "endfix": ("STRING", {"default": ""}),
                "format": (["JPG", "WEBP", "PNG"],),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "keep_alpha": ("BOOLEAN", {"default": True, "display_name": "Keep transparency"}),
                "save_png_metadata": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = "Shiro Tools/Export"

    def save_image(self, images, export_config, endfix, format, quality, keep_alpha, save_png_metadata,
                    mask=None, prompt=None, extra_pnginfo=None):
        full_path = export_config["full_path"]; prefix = export_config["prefix"]; counter = export_config["counter"]
        mask_batch = int(mask.shape[0]) if mask is not None else 0

        for idx, image in enumerate(images):
            filename = f"{prefix}{counter:05d}{endfix}.{format.lower()}"; file_path = os.path.join(full_path, filename)
            while os.path.exists(file_path):
                counter += 1; filename = f"{prefix}{counter:05d}{endfix}.{format.lower()}"; file_path = os.path.join(full_path, filename)

            frame_mask = None
            if mask is not None and mask_batch > 0:
                # se o mask tiver menos frames que as images (ex: 1 mask pra
                # um batch de imagens), reusa ciclicamente em vez de estourar
                frame_mask = mask[idx % mask_batch]

            _shiro_save_frame_alpha(
                image, frame_mask, file_path, format, quality, keep_alpha,
                save_png_metadata=save_png_metadata, prompt=prompt, extra_pnginfo=extra_pnginfo,
            )
            counter += 1
        return {}

class SaveExportA1111:
    def __init__(self): self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "output_folder": ("STRING", {"default": "extra-imagem"}),
                "prefix": ("STRING", {"default": ""}),
                "endfix": ("STRING", {"default": ""}),
                "format": (["JPG", "WEBP", "PNG"],),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "keep_alpha": ("BOOLEAN", {"default": True, "display_name": "Keep transparency"}),
                "save_png_metadata": ("BOOLEAN", {"default": False}),
                "disable_counter": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = "Shiro Tools/Export"

    def save_image(self, images, output_folder, prefix, endfix, format, quality, keep_alpha, save_png_metadata,
                    disable_counter=False, mask=None, prompt=None, extra_pnginfo=None):
        from datetime import datetime
        hoje = datetime.now().strftime("%Y-%m-%d"); output_folder = output_folder.replace("%date:yyyy-MM-dd%", hoje).replace("%date%", hoje)
        full_path = os.path.join(self.output_dir, output_folder); os.makedirs(full_path, exist_ok=True)
        if not disable_counter: counter = _shiro_calculate_next_counter(full_path)

        mask_batch = int(mask.shape[0]) if mask is not None else 0

        for idx, image in enumerate(images):
            if disable_counter:
                batch_suffix = f"_{idx}" if len(images) > 1 else ""
                filename = f"{prefix}{endfix}{batch_suffix}.{format.lower()}"; file_path = os.path.join(full_path, filename)
            else:
                filename = f"{prefix}{counter:05d}{endfix}.{format.lower()}"; file_path = os.path.join(full_path, filename)
                while os.path.exists(file_path):
                    counter += 1; filename = f"{prefix}{counter:05d}{endfix}.{format.lower()}"; file_path = os.path.join(full_path, filename)

            frame_mask = None
            if mask is not None and mask_batch > 0:
                frame_mask = mask[idx % mask_batch]

            _shiro_save_frame_alpha(
                image, frame_mask, file_path, format, quality, keep_alpha,
                save_png_metadata=save_png_metadata, prompt=prompt, extra_pnginfo=extra_pnginfo,
            )
            if not disable_counter: counter += 1
        return {}

NODE_CLASS_MAPPINGS = {
    "ShiroLoadWatermark": ShiroLoadWatermark,
    "ShiroWatermark": ShiroWatermark,
    "ShiroExportGlobalConfig": ShiroExportGlobalConfig,
    "ShiroSaveExportLinked": ShiroSaveExportLinked,
    "SaveExportA1111": SaveExportA1111,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroLoadWatermark": "Load Watermark",
    "ShiroWatermark": "Watermark",
    "ShiroExportGlobalConfig": "Export Global Config",
    "ShiroSaveExportLinked": "Export Image A1111 (Linked)",
    "SaveExportA1111": "Export Image A1111 (Legacy)",
}