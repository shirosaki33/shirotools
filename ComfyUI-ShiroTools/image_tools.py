import comfy.utils

# ---------------------------------------------------------------------------
# NODES (Imagem)
# ---------------------------------------------------------------------------

class ShiroHiresFixLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.01, "display": "number"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "round_to_multiple": ([8, 16, 32, 64], {"default": 8}),
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "pixel_width", "pixel_height")
    FUNCTION = "do_upscale"
    CATEGORY = "Shiro Tools/Image"

    def do_upscale(self, latent, scale_factor, upscale_method, round_to_multiple=8):
        l_samples = latent["samples"]
        ndim = l_samples.dim()
        if ndim == 5: b, c, t, h, w = l_samples.shape
        else: h, w = l_samples.shape[2], l_samples.shape[3]
        orig_pixel_height = h * 8; orig_pixel_width = w * 8
        raw_width = orig_pixel_width * scale_factor; raw_height = orig_pixel_height * scale_factor
        m = round_to_multiple
        pixel_width = max(m, int(round(raw_width / m)) * m); pixel_height = max(m, int(round(raw_height / m)) * m)
        l_new_width = pixel_width // 8; l_new_height = pixel_height // 8
        if ndim == 5:
            samples_4d = l_samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            upscaled_4d = comfy.utils.common_upscale(samples_4d, l_new_width, l_new_height, upscale_method, "disabled")
            upscaled_latent = upscaled_4d.reshape(b, t, c, l_new_height, l_new_width).permute(0, 2, 1, 3, 4)
        else:
            upscaled_latent = comfy.utils.common_upscale(l_samples, l_new_width, l_new_height, upscale_method, "disabled")
        new_latent = latent.copy()
        new_latent["samples"] = upscaled_latent

        if "noise_mask" in latent:
            mask = latent["noise_mask"]
            orig_ndim = len(mask.shape)

            if orig_ndim == 3:
                mask4d = mask.unsqueeze(1)
            elif orig_ndim == 2:
                mask4d = mask.unsqueeze(0).unsqueeze(0)
            else:
                mask4d = mask

            upscaled_mask = comfy.utils.common_upscale(mask4d, l_new_width, l_new_height, "bilinear", "disabled")

            if orig_ndim == 3:
                new_latent["noise_mask"] = upscaled_mask.squeeze(1)
            elif orig_ndim == 2:
                new_latent["noise_mask"] = upscaled_mask.squeeze(0).squeeze(0)
            else:
                new_latent["noise_mask"] = upscaled_mask

        return (new_latent, pixel_width, pixel_height)

class ShiroHiresFixImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.01, "display": "number"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic" ,"lanczos"],),
                "round_to_multiple": ([8, 16, 32, 64], {"default": 8}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "do_upscale"
    CATEGORY = "Shiro Tools/Image"

    def do_upscale(self, image, scale_factor, upscale_method, round_to_multiple=8):
        orig_width = image.shape[2]; orig_height = image.shape[1]
        raw_width = orig_width * scale_factor; raw_height = orig_height * scale_factor
        m = round_to_multiple
        pixel_width = max(m, int(round(raw_width / m)) * m); pixel_height = max(m, int(round(raw_height / m)) * m)
        img_samples = image.movedim(-1, 1)
        upscaled_img = comfy.utils.common_upscale(img_samples, pixel_width, pixel_height, upscale_method, "disabled")
        return (upscaled_img.movedim(1, -1), pixel_width, pixel_height)

NODE_CLASS_MAPPINGS = {
    "ShiroHiresFixLatent": ShiroHiresFixLatent,
    "ShiroHiresFixImage": ShiroHiresFixImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroHiresFixLatent": "HiresFix Latent",
    "ShiroHiresFixImage": "HiresFix Image",
}