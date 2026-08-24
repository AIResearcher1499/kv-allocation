"""Command line entry point (pattern from the fertility-precision repo)."""

from __future__ import annotations

import sys


def doctor() -> int:
    """Import everything a run needs, then report the accelerator.

    Runs in seconds; exists so a missing/broken install fails HERE and not
    twenty minutes into a grid shard.
    """
    import importlib

    ok = True
    for mod in ("torch",):
        try:
            m = importlib.import_module(mod)
            print(f"  ok      {mod:<10} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  MISSING {mod:<10} {type(exc).__name__}: {exc}")
    if ok:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  gpu     cuda:{i} {p.name} {p.total_memory // 2**20} MiB")
        elif torch.backends.mps.is_available():
            print("  gpu     mps (dev box — grids belong on the A6000s)")
        else:
            print("  gpu     NONE (cpu only)")
        # exercise the exact kernel path the model uses
        import torch.nn.functional as F

        q = torch.randn(1, 2, 4, 8)
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        print("  ok      sdpa causal path")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: kvalloc {doctor|a0} [args...]\n"
              "  doctor  check installs + GPU before spending any compute\n"
              "  a0      Stage A-0 runner (see `kvalloc a0 --help`)")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "doctor":
        return doctor()
    if cmd == "a0":
        from .a0 import main as a0_main

        a0_main(rest)
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
