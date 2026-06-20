"""Prime-RL launch compatibility hooks.

Python imports sitecustomize automatically when this directory is on PYTHONPATH.
Keep this small: it only forces Prime-RL's own upstream-compat patches to run
before third-party imports in launcher entrypoints.
"""

try:
    import prime_rl._compat  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "prime_rl":
        raise
