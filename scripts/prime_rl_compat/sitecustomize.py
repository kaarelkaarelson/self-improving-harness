"""Prime-RL launch compatibility hooks.

Python imports sitecustomize automatically when this directory is on PYTHONPATH.
Keep this small and self-contained so launcher entrypoints get patched before
third-party imports, even before Prime-RL's editable package is importable.
"""

try:
    import transformers.modeling_flash_attention_utils as _mfau
except ModuleNotFoundError:
    _mfau = None

if _mfau is not None and not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
    # ring_flash_attn 0.1.8 imports this symbol from the old location.
    # Transformers 5.x still exposes the helper in transformers.utils.
    try:
        from transformers.utils import is_flash_attn_greater_or_equal_2_10 as _is_flash_attn_ge_2_10
    except ImportError:
        _is_flash_attn_ge_2_10 = lambda: True

    _mfau.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_ge_2_10
