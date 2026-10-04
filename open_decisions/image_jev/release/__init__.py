"""Fallback release of the frozen base model with route-bound temperature calibration.

``predictor`` is the request contract and readout, ``server`` the HTTP wrapper,
``bundle`` the release directory generator, ``measure_body`` the board-style
measurement manifest body builder.  Only ``predictor``, ``server`` and ``smoke``
are shipped inside the release bundle.
"""
