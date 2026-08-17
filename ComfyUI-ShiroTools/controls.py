# ---------------------------------------------------------------------------
# NODES (Flow Control e Switches)
# ---------------------------------------------------------------------------

class ShiroBooleanValidator:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {}, "optional": {"anything": ("*",)},}
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "validate"
    CATEGORY = "Shiro Tools/Switches"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs): return float("nan")

    def validate(self, anything=None):
        if isinstance(anything, bool): return (anything,)
        return (False,)

class _ShiroStageSwitch8Base:
    TYPE_NAME = "*"
    SLOT_PREFIX = "anything"
    NUM_SLOTS = 8
    START_VALUE = 0
    STEP = 1

    @classmethod
    def INPUT_TYPES(cls):
        required = {f"{cls.SLOT_PREFIX}_1": (cls.TYPE_NAME,),}
        optional = {}
        for i in range(2, cls.NUM_SLOTS + 1):
            optional[f"{cls.SLOT_PREFIX}_{i}"] = (cls.TYPE_NAME,)
            optional[f"enabled_{i}"] = ("BOOLEAN", {"default": True})
        return {"required": required, "optional": optional}
    FUNCTION = "switch"
    CATEGORY = "Shiro Tools/Switches"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs): return float("nan")

    def switch(self, **kwargs):
        prefix = self.SLOT_PREFIX; n = self.NUM_SLOTS
        values = [kwargs.get(f"{prefix}_{i}") for i in range(1, n + 1)]
        enabled_flags = [True] + [bool(kwargs.get(f"enabled_{i}", True)) for i in range(2, n + 1)]
        selected_value = values[0]; selected_index = 0
        for i in range(n - 1, 0, -1):
            if enabled_flags[i] and values[i] is not None:
                selected_value = values[i]; selected_index = i
                break
        index = self.START_VALUE + self.STEP * selected_index
        return (selected_value, int(index))

class ShiroStageSwitch8Latent(_ShiroStageSwitch8Base):
    TYPE_NAME = "LATENT"; SLOT_PREFIX = "latent"
    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "index")

class ShiroStageSwitch8Image(_ShiroStageSwitch8Base):
    TYPE_NAME = "IMAGE"; SLOT_PREFIX = "image"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "index")

class ShiroStageSwitch8Model(_ShiroStageSwitch8Base):
    TYPE_NAME = "MODEL"; SLOT_PREFIX = "model"
    RETURN_TYPES = ("MODEL", "INT")
    RETURN_NAMES = ("model", "index")

class ShiroStageSwitch8Any(_ShiroStageSwitch8Base):
    TYPE_NAME = "*"; SLOT_PREFIX = "anything"
    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("anything", "index")

NODE_CLASS_MAPPINGS = {
    "ShiroBooleanValidator": ShiroBooleanValidator,
    "ShiroStageSwitch8Latent": ShiroStageSwitch8Latent,
    "ShiroStageSwitch8Image": ShiroStageSwitch8Image,
    "ShiroStageSwitch8Model": ShiroStageSwitch8Model,
    "ShiroStageSwitch8Any": ShiroStageSwitch8Any,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroBooleanValidator": "Boolean Validator",
    "ShiroStageSwitch8Latent": "Stage Switch 8 - LATENT",
    "ShiroStageSwitch8Image": "Stage Switch 8 - IMAGE",
    "ShiroStageSwitch8Model": "Stage Switch 8 - MODEL",
    "ShiroStageSwitch8Any": "Stage Switch 8 - ANY",
}