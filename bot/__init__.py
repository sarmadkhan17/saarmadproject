"""CryptoBot v5 package.

The bot runs with this directory on sys.path (see launcher.py / dashboard),
so modules are imported directly by sub-package, e.g. `from engine.futures
import FuturesBot`. This file intentionally performs no re-exports: the old
v3 compatibility shims pointed at the removed ML stack (models.*, tuning.learner,
TrainingFeed) and were dead/broken.
"""
