import torch

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (Áudio)
# ---------------------------------------------------------------------------

def _audio_has_data(audio):
    if not isinstance(audio, dict): return False
    waveform = audio.get("waveform", None)
    if waveform is None or not torch.is_tensor(waveform) or waveform.numel() == 0: return False
    return True

def _normalize_waveform_shape(waveform):
    original_dim = waveform.dim()
    if original_dim == 1: work = waveform.unsqueeze(0).unsqueeze(0)
    elif original_dim == 2: work = waveform.unsqueeze(0)
    elif original_dim == 3: work = waveform
    else: return None, original_dim
    return work, original_dim

def _restore_waveform_shape(work, original_dim):
    if original_dim == 1: return work.squeeze(0).squeeze(0)
    if original_dim == 2: return work.squeeze(0)
    return work

def _per_item_envelope(work):
    """Calcula o envelope de amplitude por item do batch (colapsando só o
    canal, nunca o batch). work: [B, C, T] -> retorna [B, T].

    Isso evita o bug de usar amax sobre (batch, canal) juntos, que faz um
    item barulhento do lote "mascarar" o silêncio de outro item silencioso
    (ou vice-versa), gerando cortes errados quando o AUDIO tiver mais de
    um clipe real no batch (batch_size > 1 com conteúdos diferentes, e
    não só canais estéreo do mesmo clipe)."""
    return work.abs().amax(dim=1)

# ---------------------------------------------------------------------------
# NODES (Áudio)
# ---------------------------------------------------------------------------

class ShiroTrimLeadingSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": 0.0, "step": 1.0, "display": "slider"}),
                "min_silence_ms": ("INT", {"default": 250, "min": 0, "max": 5000, "step": 10}),
                "padding_ms": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 10}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim"
    CATEGORY = "Shiro Tools/Audio"

    def trim(self, audio, threshold_db=-45.0, min_silence_ms=250, padding_ms=50):
        if not isinstance(audio, dict) or "waveform" not in audio: return (audio,)
        waveform = audio["waveform"]; sample_rate = int(audio.get("sample_rate", 44100))
        if waveform is None or not torch.is_tensor(waveform) or waveform.numel() == 0: return (audio,)
        work, original_dim = _normalize_waveform_shape(waveform)
        if work is None: return (audio,)
        threshold = float(10.0 ** (float(threshold_db) / 20.0))
        envelope = _per_item_envelope(work)  # [B, T], nunca mistura itens do batch
        b = envelope.shape[0]
        first_active_per_item = []
        for i in range(b):
            active_indices = torch.nonzero(envelope[i] > threshold, as_tuple=False)
            # sem atividade detectada nesse item: não sabemos onde ele "começa",
            # então não usamos esse item pra limitar o corte comum.
            if active_indices.numel() > 0:
                first_active_per_item.append(int(active_indices[0].item()))
        if not first_active_per_item: return (audio,)
        # usa o MENOR first_active entre os itens: garante que o corte comum
        # aplicado ao batch inteiro nunca remova áudio real de nenhum item.
        first_active = min(first_active_per_item)
        min_silence_samples = int(sample_rate * (int(min_silence_ms) / 1000.0))
        if first_active < min_silence_samples: return (audio,)
        padding_samples = int(sample_rate * (int(padding_ms) / 1000.0))
        start = max(0, first_active - padding_samples)
        out_audio = dict(audio); trimmed = work[..., start:]
        out_audio["waveform"] = _restore_waveform_shape(trimmed, original_dim).contiguous()
        out_audio["sample_rate"] = sample_rate
        return (out_audio,)

class ShiroLimitLongSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": 0.0, "step": 1.0, "display": "slider"}),
                "min_silence_ms": ("INT", {"default": 350, "min": 20, "max": 5000, "step": 10}),
                "max_silence_ms": ("INT", {"default": 180, "min": 0, "max": 2000, "step": 10}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "limit"
    CATEGORY = "Shiro Tools/Audio"

    def limit(self, audio, threshold_db=-45.0, min_silence_ms=350, max_silence_ms=180):
        if not isinstance(audio, dict) or "waveform" not in audio: return (audio,)
        waveform = audio["waveform"]; sample_rate = int(audio.get("sample_rate", 44100))
        if waveform is None or not torch.is_tensor(waveform) or waveform.numel() == 0: return (audio,)
        work, original_dim = _normalize_waveform_shape(waveform)
        if work is None: return (audio,)
        num_samples = work.shape[-1]; threshold = float(10.0 ** (float(threshold_db) / 20.0))
        envelope = _per_item_envelope(work)  # [B, T], nunca mistura itens do batch
        # um instante só conta como "silêncio" se TODOS os itens do batch
        # estiverem silenciosos ali; assim o corte comum aplicado ao
        # tensor inteiro nunca remove áudio real de nenhum item.
        silent = (envelope <= threshold).all(dim=0)
        min_silence_samples = int(sample_rate * (int(min_silence_ms) / 1000.0))
        max_silence_samples = int(sample_rate * (int(max_silence_ms) / 1000.0))
        if min_silence_samples <= 0: return (audio,)
        cuts = []; pos = 0
        while pos < num_samples:
            while pos < num_samples and not bool(silent[pos].item()): pos += 1
            if pos >= num_samples: break
            start = pos
            while pos < num_samples and bool(silent[pos].item()): pos += 1
            end = pos; length = end - start
            if length >= min_silence_samples and length > max_silence_samples: cuts.append((start + max_silence_samples, end))
        if not cuts: return (audio,)
        pieces = []; cursor = 0
        for cut_start, cut_end in cuts:
            if cursor < cut_start: pieces.append(work[..., cursor:cut_start])
            cursor = cut_end
        if cursor < num_samples: pieces.append(work[..., cursor:])
        if not pieces: return (audio,)
        out_audio = dict(audio); shortened = torch.cat(pieces, dim=-1)
        out_audio["waveform"] = _restore_waveform_shape(shortened, original_dim).contiguous()
        out_audio["sample_rate"] = sample_rate
        return (out_audio,)

class ShiroAudioSelector8:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "selected_slot": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "fallback_to_audio_1": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio_2": ("AUDIO", {"lazy": True}), "audio_3": ("AUDIO", {"lazy": True}),
                "audio_4": ("AUDIO", {"lazy": True}), "audio_5": ("AUDIO", {"lazy": True}),
                "audio_6": ("AUDIO", {"lazy": True}), "audio_7": ("AUDIO", {"lazy": True}),
                "audio_8": ("AUDIO", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "select"
    CATEGORY = "Shiro Tools/Audio"

    def select(self, audio_1, selected_slot=1, fallback_to_audio_1=True, audio_2=None, audio_3=None, audio_4=None, audio_5=None, audio_6=None, audio_7=None, audio_8=None):
        slots = {1: audio_1, 2: audio_2, 3: audio_3, 4: audio_4, 5: audio_5, 6: audio_6, 7: audio_7, 8: audio_8}
        try: slot = int(selected_slot)
        except Exception: slot = 1
        selected = slots.get(slot)
        if _audio_has_data(selected): return (selected,)
        if bool(fallback_to_audio_1): return (audio_1,)
        for i in range(1, 9):
            candidate = slots.get(i)
            if _audio_has_data(candidate): return (candidate,)
        return (audio_1,)

NODE_CLASS_MAPPINGS = {
    "ShiroTrimLeadingSilence": ShiroTrimLeadingSilence,
    "ShiroLimitLongSilence": ShiroLimitLongSilence,
    "ShiroAudioSelector8": ShiroAudioSelector8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShiroTrimLeadingSilence": "Trim Leading Silence",
    "ShiroLimitLongSilence": "Limit Long Silence",
    "ShiroAudioSelector8": "Audio Selector 8 Slots",
}