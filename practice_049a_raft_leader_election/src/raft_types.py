"""Re-export all types from 00_raft_types.py for clean imports.

Other modules import from here:
    from src.raft_types import NodeState, LogEntry, RaftConfig, ...

This avoids importing from '00_raft_types' which has a leading digit
(valid Python but awkward).
"""

import importlib.util
import sys
from pathlib import Path

# Ensure the practice root is on sys.path
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

# Import the actual module by file path (because of leading digit in name).
# Register in sys.modules so dataclasses can resolve __module__.
_MODULE_NAME = "src._raft_types_impl"
if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        Path(__file__).resolve().parent / "00_raft_types.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules[_MODULE_NAME]

# Re-export everything
NodeState = _mod.NodeState
LogEntry = _mod.LogEntry
RequestVoteArgs = _mod.RequestVoteArgs
RequestVoteReply = _mod.RequestVoteReply
AppendEntriesArgs = _mod.AppendEntriesArgs
AppendEntriesReply = _mod.AppendEntriesReply
RaftConfig = _mod.RaftConfig

__all__ = [
    "NodeState",
    "LogEntry",
    "RequestVoteArgs",
    "RequestVoteReply",
    "AppendEntriesArgs",
    "AppendEntriesReply",
    "RaftConfig",
]
