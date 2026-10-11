# tests/fixtures/synth.py — backward-compat re-export shim.
#
# Canonical home is exa_home.synth (prod code must not import test paths).
# This module re-exports the same names so existing test imports keep
# working unchanged.
from exa_home.synth import make_docs, make_vectors, make_wet_gz

__all__ = ["make_docs", "make_vectors", "make_wet_gz"]
