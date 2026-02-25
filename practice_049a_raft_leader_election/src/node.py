"""Re-export RaftNode and create_cluster from 01_node_state_machine.py.

Other modules import from here:
    from src.node import RaftNode, create_cluster

This avoids importing from '01_node_state_machine' which has a leading digit.
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_NAME = "src._node_state_machine_impl"
if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        Path(__file__).resolve().parent / "01_node_state_machine.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules[_MODULE_NAME]

RaftNode = _mod.RaftNode
create_cluster = _mod.create_cluster

__all__ = ["RaftNode", "create_cluster"]
