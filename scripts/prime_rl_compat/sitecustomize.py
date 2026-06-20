"""Prime-RL launch compatibility hooks.

Python imports sitecustomize automatically when this directory is on PYTHONPATH.
Keep this small and self-contained so launcher entrypoints get patched before
third-party imports, even before Prime-RL's editable package is importable.
"""

from pathlib import Path
import sys

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


def _ensure_prime_rl_src_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    prime_rl_src = repo_root / "prime-rl" / "src"
    if prime_rl_src.exists():
        sys.path.insert(0, str(prime_rl_src))


def _patch_fused_ce_zero_token_shards() -> None:
    """Avoid NaN loss on CP shards whose labels are all ignore_index.

    Under context parallelism, a shard can contain no trainable SFT tokens even
    though the full sequence does. Liger's mean-reduction fused CE returns NaN
    when every local label is ignore_index. Compute a dummy local CE and multiply
    by zero instead, preserving graph/collective participation with zero grad.
    """

    try:
        import torch
        from prime_rl.trainer.models.layers.lm_head import FusedCrossEntropyOutputLinear, PrimeLmOutput
    except ModuleNotFoundError:
        return

    if getattr(FusedCrossEntropyOutputLinear, "_sih_zero_token_patch", False):
        return

    original_forward = FusedCrossEntropyOutputLinear.forward

    def patched_forward(self, hidden_states, labels=None, temperature=None):
        if labels is not None and torch.all(labels == self.IGNORE_INDEX):
            dummy_labels = labels.clone()
            dummy_labels.reshape(-1)[0] = 0
            output = original_forward(self, hidden_states, dummy_labels, temperature)
            return PrimeLmOutput(loss=output["loss"] * 0.0)
        return original_forward(self, hidden_states, labels, temperature)

    FusedCrossEntropyOutputLinear.forward = patched_forward
    FusedCrossEntropyOutputLinear._sih_zero_token_patch = True


_ensure_prime_rl_src_importable()
_patch_fused_ce_zero_token_shards()
