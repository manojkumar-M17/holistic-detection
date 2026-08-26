"""
conftest.py — Project-root pytest configuration.

Placing this file here causes pytest to automatically insert the project root
(m:\\holistic detection) into sys.path, so that all test files can import
project packages (e.g. `from modules.object_tracker import ObjectTracker`)
without needing to modify PYTHONPATH manually or add sys.path hacks.
"""
