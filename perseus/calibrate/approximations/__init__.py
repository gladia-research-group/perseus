"""Built-in FHE approximations. Importing this package registers them all.

Each module owns one op kind end-to-end — site matcher, sample collector, and
fitter — registered via `perseus.calibrate.registry.register`. Adding support
for a new nonlinearity is one new module here; nothing else changes.
"""

from perseus.calibrate.approximations import cutmax, gelu, norm, softmax  # noqa: F401
