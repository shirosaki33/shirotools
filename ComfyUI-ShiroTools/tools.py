import time

# ---------------------------------------------------------------------------
# NODES (Tools, Timing e Math)
# ---------------------------------------------------------------------------

class ShiroImageDelay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "delay_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "always_run": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "delay"
    CATEGORY = "Shiro Tools/Tools"

    @classmethod
    def IS_CHANGED(cls, image, delay_seconds=5.0, always_run=True):
        if bool(always_run): return time.time()
        return f"{float(delay_seconds):.3f}"

    def delay(self, image, delay_seconds=5.0, always_run=True):
        seconds = max(0.0, float(delay_seconds))
        if seconds > 0.0: time.sleep(seconds)
        return (image,)

class ShiroPassthroughDelay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*",),
                "delay_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "always_run": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("anything",)
    FUNCTION = "delay"
    CATEGORY = "Shiro Tools/Tools"

    @classmethod
    def IS_CHANGED(cls, anything, delay_seconds=5.0, always_run=True):
        if bool(always_run): return time.time()
        return f"{float(delay_seconds):.3f}"

    def delay(self, anything, delay_seconds=5.0, always_run=True):
        seconds = max(0.0, float(delay_seconds))
        if seconds > 0.0: time.sleep(seconds)
        return (anything,)

class ShiroFloatPrecise2:
    """FLOAT simples, sempre arredondado/exibido com exatamente 2 casas decimais."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.00,
                    "min": -999999.00,
                    "max": 999999.00,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number",
                }),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = "Shiro Tools/Tools"

    def get_value(self, value):
        rounded = round(float(value), 2)
        return (rounded,)

class StandaloneResolutionScaler:
    """Resolution Scaler Simples Original"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("scaled_width", "scaled_height", "scale_factor")
    FUNCTION = "calculate"
    CATEGORY = "Shiro Tools/Tools" 

    def calculate(self, image, scale_factor):
        _, h, w, _ = image.shape
        final_w = int(w * scale_factor)
        final_h = int(h * scale_factor)
        return (final_w, final_h, scale_factor)

class ShiroResolutionScalerPrecise:
    """Resolution Scaler ajustado para suportar até 3 casas decimais e modo divisão (Plano B)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.001,
                    "round": 0.001,
                }),
                "divide_by": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0001,
                    "max": 1000.0,
                    "step": 0.001,
                    "round": 0.001,
                    "display_name": "Divide scale factor by (plano B)",
                }),
            }
        }
    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("scaled_width", "scaled_height", "effective_scale_factor")
    FUNCTION = "calculate"
    CATEGORY = "Shiro Tools/Tools" 

    def calculate(self, image, scale_factor, divide_by=1.0):
        divisor = float(divide_by) if abs(float(divide_by)) > 1e-9 else 1.0
        effective_scale = float(scale_factor) / divisor
        _, h, w, _ = image.shape
        final_w = int(w * effective_scale)
        final_h = int(h * effective_scale)
        return (final_w, final_h, float(effective_scale))

class ShiroResolutionScalerLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("scaled_width", "scaled_height", "scale_factor")
    FUNCTION = "calculate"
    CATEGORY = "Shiro Tools/Tools"

    def calculate(self, latent, scale_factor):
        l_samples = latent["samples"]
        h, w = l_samples.shape[-2], l_samples.shape[-1]
        base_w = w * 8; base_h = h * 8
        final_w = int(base_w * scale_factor); final_h = int(base_h * scale_factor)
        return (final_w, final_h, scale_factor)

class ShiroAdvancedDenoiseMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_to_run": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("steps", "start_at_step", "end_at_step")
    FUNCTION = "calc_steps"
    CATEGORY = "Shiro Tools/Tools"

    def calc_steps(self, steps_to_run, denoise):
        if denoise <= 0.0: return (steps_to_run, steps_to_run, steps_to_run)
        total_steps = max(1, int(steps_to_run / max(0.01, denoise)))
        start_at_step = total_steps - steps_to_run
        return (total_steps, start_at_step, total_steps)

NODE_CLASS_MAPPINGS = {
    "ShiroImageDelay": ShiroImageDelay,
    "ShiroPassthroughDelay": ShiroPassthroughDelay,
    "ShiroFloatPrecise2": ShiroFloatPrecise2,
    "StandaloneResolutionScaler": StandaloneResolutionScaler,
    "ShiroResolutionScalerPrecise": ShiroResolutionScalerPrecise,
    "ShiroResolutionScalerLatent": ShiroResolutionScalerLatent,
    "ShiroAdvancedDenoiseMath": ShiroAdvancedDenoiseMath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroImageDelay": "Image Workflow Delay",
    "ShiroPassthroughDelay": "Passthrough Workflow Delay",
    "ShiroFloatPrecise2": "Float 2 Decimals",
    "StandaloneResolutionScaler": "Resolution Scaler (Image)",
    "ShiroResolutionScalerPrecise": "Resolution Scaler Precise (Image)",
    "ShiroResolutionScalerLatent": "Resolution Scaler (Latent)",
    "ShiroAdvancedDenoiseMath": "Advanced Denoise Math",
}