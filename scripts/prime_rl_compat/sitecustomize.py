"""Prime-RL launch compatibility hooks.

Python imports sitecustomize automatically when this directory is on PYTHONPATH.
Keep this small and self-contained so launcher entrypoints get patched before
third-party imports, even before Prime-RL's editable package is importable.
"""

from pathlib import Path
import importlib.abc
import importlib.machinery
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


def _apply_fused_ce_zero_token_patch(module) -> None:
    """Avoid NaN loss on CP shards whose labels are all ignore_index.

    Under context parallelism, a shard can contain no trainable SFT tokens even
    though the full sequence does. Liger's mean-reduction fused CE returns NaN
    when every local label is ignore_index. Compute a dummy local CE and multiply
    by zero instead, preserving graph/collective participation with zero grad.
    """

    FusedCrossEntropyOutputLinear = module.FusedCrossEntropyOutputLinear
    PrimeLmOutput = module.PrimeLmOutput
    torch = module.torch

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


class _FusedCEPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self.wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        _apply_fused_ce_zero_token_patch(module)


class _FusedCEPatchFinder(importlib.abc.MetaPathFinder):
    TARGET = "prime_rl.trainer.models.layers.lm_head"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _FusedCEPatchLoader(spec.loader)
        return spec


def _install_fused_ce_patch_hook() -> None:
    target = _FusedCEPatchFinder.TARGET
    module = sys.modules.get(target)
    if module is not None:
        _apply_fused_ce_zero_token_patch(module)
        return
    if not any(isinstance(finder, _FusedCEPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FusedCEPatchFinder())


_ensure_prime_rl_src_importable()
_install_fused_ce_patch_hook()
