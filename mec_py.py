"""
MEC - Master Engine Control
Python Glue Layer

This is the link between the C core and Python-based engines.
It allows Python code to:
  - Register as an engine brick
  - Receive tasks from the baseplate
  - Report health back to the baseplate
  - Call back into the C core

LEGO PHILOSOPPY: Python engines are just another type of brick.
"""

import ctypes
import os
import sys
import json
import time
import threading
import importlib
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum


# ── Constants (must match C headers) ───────────────────────────────
class EngineType(IntEnum):
    MODEL = 0
    ROUTER = 1
    DECOMPILER = 2
    SECURITY = 3
    SELF_HEALER = 4
    CUSTOM = 100


class EngineState(IntEnum):
    UNINITIALIZED = 0
    INITIALIZING = 1
    ACTIVE = 2
    DEGRADED = 3
    FAILED = 4
    REPAIRING = 5
    STOPPING = 6
    STOPPED = 7
    REMOVED = 8


class HealthStatus(IntEnum):
    UNKNOWN = 0
    HEALTHY = 1
    DEGRADED = 2
    CRITICAL = 3
    UNRESPONSIVE = 4


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# ── Data Classes ───────────────────────────────────────────────────
@dataclass
class Task:
    task_id: int
    capability: str
    input_data: bytes = b""
    priority: Priority = Priority.NORMAL
    status: int = 0  # 0=pending, 1=running, 2=done, -1=failed
    output_data: bytes = b""
    assigned_engine: str = ""
    retry_count: int = 0


@dataclass
class Health:
    status: HealthStatus = HealthStatus.UNKNOWN
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tasks_processed: int = 0
    tasks_failed: int = 0
    uptime_seconds: int = 0
    message: str = ""


@dataclass
class Model:
    name: str
    path: str
    version: str = "1.0.0"
    loaded: bool = False
    active: bool = False


# ── Python Engine Base Class ───────────────────────────────────────
class PythonEngine:
    """
    Base class for Python-based engine bricks.
    
    Subclass this and implement the required methods.
    The C baseplate will call these methods through the glue layer.
    """

    def __init__(self, name: str, engine_type: EngineType = EngineType.CUSTOM):
        self.name = name
        self.engine_type = engine_type
        self.state = EngineState.UNINITIALIZED
        self.health = Health()
        self.capabilities: List[str] = []
        self.private_data: Dict[str, Any] = {}
        self._running = False
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────
    def init(self, config: Dict[str, Any]) -> int:
        """Initialize the brick. Return 0 on success."""
        self.state = EngineState.INITIALIZING
        self.private_data["config"] = config
        return 0

    def start(self) -> int:
        """Start the brick. Return 0 on success."""
        self._running = True
        self.state = EngineState.ACTIVE
        self.health.status = HealthStatus.HEALTHY
        return 0

    def stop(self) -> int:
        """Stop the brick. Return 0 on success."""
        self._running = False
        self.state = EngineState.STOPPED
        return 0

    def destroy(self) -> None:
        """Clean up resources."""
        pass

    # ── Work ───────────────────────────────────────────────────────
    def submit_task(self, task: Task) -> int:
        """Process a task. Return 0 on success."""
        try:
            result = self.process_task(task)
            task.output_data = result
            task.status = 2  # done
            self.health.tasks_processed += 1
            return 0
        except Exception as e:
            task.status = -1  # failed
            self.health.tasks_failed += 1
            self.health.message = str(e)
            return -1

    def cancel_task(self, task_id: int) -> int:
        """Cancel a running task. Return 0 on success."""
        return 0

    def process_task(self, task: Task) -> bytes:
        """
        Override this to implement actual work.
        Must return bytes as output.
        """
        return b""

    # ── Health ─────────────────────────────────────────────────────
    def health_check(self) -> Health:
        """Return current health status."""
        return self.health

    def repair(self, action: str) -> int:
        """Attempt self-repair. Return 0 on success."""
        self.state = EngineState.REPAIRING
        # Default: just restart
        self.stop()
        self.start()
        self.state = EngineState.ACTIVE
        return 0

    # ── Model Management ───────────────────────────────────────────
    def load_model(self, model_path: str) -> int:
        """Load a model. Return 0 on success."""
        return 0

    def unload_model(self, model_name: str) -> int:
        """Unload a model. Return 0 on success."""
        return 0

    def switch_model(self, model_name: str) -> int:
        """Switch to a different model. Return 0 on success."""
        return 0

    # ── Capabilities ──────────────────────────────────────────────
    def can_handle(self, capability: str) -> bool:
        """Return True if this brick can handle the given capability."""
        return capability in self.capabilities

    def add_capability(self, capability: str) -> None:
        """Advertise a capability."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)


# ── Engine Registry ────────────────────────────────────────────────
class EngineRegistry:
    """Manages all Python engine bricks."""

    def __init__(self):
        self._engines: Dict[str, PythonEngine] = {}
        self._lock = threading.Lock()

    def register(self, engine: PythonEngine) -> int:
        with self._lock:
            if engine.name in self._engines:
                return -1
            self._engines[engine.name] = engine
            return 0

    def unregister(self, name: str) -> int:
        with self._lock:
            if name not in self._engines:
                return -1
            engine = self._engines[name]
            engine.stop()
            engine.destroy()
            del self._engines[name]
            return 0

    def get(self, name: str) -> Optional[PythonEngine]:
        return self._engines.get(name)

    def get_by_type(self, engine_type: EngineType) -> Optional[PythonEngine]:
        for engine in self._engines.values():
            if engine.engine_type == engine_type and engine.state == EngineState.ACTIVE:
                return engine
        return None

    def get_by_capability(self, capability: str) -> Optional[PythonEngine]:
        for engine in self._engines.values():
            if engine.state == EngineState.ACTIVE and engine.can_handle(capability):
                return engine
        return None

    def list_engines(self) -> List[str]:
        return list(self._engines.keys())


# ── Global Registry ────────────────────────────────────────────────
_registry = EngineRegistry()


def register_engine(engine: PythonEngine) -> int:
    """Register a Python engine brick with the baseplate."""
    return _registry.register(engine)


def unregister_engine(name: str) -> int:
    """Unregister a Python engine brick."""
    return _registry.unregister(name)


def get_engine(name: str) -> Optional[PythonEngine]:
    """Get a registered engine by name."""
    return _registry.get(name)


# ── Dynamic Engine Loader ──────────────────────────────────────────
def load_engine_from_module(module_path: str, class_name: str,
                            config: Dict[str, Any]) -> Optional[PythonEngine]:
    """
    Dynamically load a Python engine from a module.
    
    Args:
        module_path: Python module path (e.g., 'engines.my_engine')
        class_name: Class name to instantiate
        config: Configuration dict to pass to the engine
    
    Returns:
        Instantiated PythonEngine or None on failure
    """
    try:
        spec = importlib.util.spec_from_file_location(module_path, f"{module_path}.py")
        if spec is None:
            # Try as package module
            module = importlib.import_module(module_path)
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        engine_class = getattr(module, class_name)
        engine = engine_class()
        engine.init(config)
        return engine
    except Exception as e:
        print(f"Failed to load engine from {module_path}: {e}", file=sys.stderr)
        return None


# ── Link to C Core ───────────────────────────────────────────────
class MECBridge:
    """
    Link to the C core via ctypes.
    This loads the shared library and provides Python-friendly wrappers.
    """

    def __init__(self, lib_path: str = "./libmec.so"):
        self._lib = None
        try:
            self._lib = ctypes.CDLL(lib_path)
            self._setup_functions()
        except OSError:
            print(f"Warning: Could not load C library at {lib_path}", file=sys.stderr)

    def _setup_functions(self):
        if not self._lib:
            return

        # mec_create
        self._lib.mec_create.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self._lib.mec_create.restype = ctypes.c_void_p

        # mec_start
        self._lib.mec_start.argtypes = [ctypes.c_void_p]
        self._lib.mec_start.restype = ctypes.c_int

        # mec_stop
        self._lib.mec_stop.argtypes = [ctypes.c_void_p]
        self._lib.mec_stop.restype = ctypes.c_int

        # mec_destroy
        self._lib.mec_destroy.argtypes = [ctypes.c_void_p]
        self._lib.mec_destroy.restype = ctypes.c_int

    def create(self, mode: int, config_path: str) -> Optional[int]:
        if not self._lib:
            return None
        result = self._lib.mec_create(mode, config_path.encode() if config_path else None)
        return result if result else None

    def start(self, ctx: int) -> int:
        if not self._lib:
            return -1
        return self._lib.mec_start(ctypes.c_void_p(ctx))

    def stop(self, ctx: int) -> int:
        if not self._lib:
            return -1
        return self._lib.mec_stop(ctypes.c_void_p(ctx))

    def destroy(self, ctx: int) -> int:
        if not self._lib:
            return -1
        return self._lib.mec_destroy(ctypes.c_void_p(ctx))


# ── Convenience Functions ──────────────────────────────────────────
def create_default_engines() -> List[PythonEngine]:
    """Create the default set of engine bricks."""
    engines = []

    # Model engine
    model_eng = PythonEngine("model_engine", EngineType.MODEL)
    model_eng.add_capability("inference")
    model_eng.add_capability("model_load")
    model_eng.add_capability("model_switch")
    engines.append(model_eng)

    # Router
    router_eng = PythonEngine("router", EngineType.ROUTER)
    router_eng.add_capability("routing")
    router_eng.add_capability("load_balance")
    router_eng.add_capability("failover")
    engines.append(router_eng)

    # Decompiler
    decomp_eng = PythonEngine("decompiler", EngineType.DECOMPILER)
    decomp_eng.add_capability("decompile")
    decomp_eng.add_capability("train")
    decomp_eng.add_capability("rebuild")
    engines.append(decomp_eng)

    # Security
    sec_eng = PythonEngine("security", EngineType.SECURITY)
    sec_eng.add_capability("defense")
    sec_eng.add_capability("offense")
    sec_eng.add_capability("scan")
    sec_eng.add_capability("patch")
    engines.append(sec_eng)

    # Self-healer
    healer_eng = PythonEngine("self_healer", EngineType.SELF_HEALER)
    healer_eng.add_capability("heal")
    healer_eng.add_capability("diagnose")
    healer_eng.add_capability("repair")
    healer_eng.add_capability("replace")
    engines.append(healer_eng)

    return engines


# ── Main Entry Point ───────────────────────────────────────────────
def main():
    """Example: Create and start the baseplate with default engines."""
    print("MEC Python Glue Layer")
    print("=" * 40)

    # Create engines
    engines = create_default_engines()
    for eng in engines:
        register_engine(eng)
        print(f"  Registered: {eng.name} (type={eng.engine_type.name}, "
              f"capabilities={eng.capabilities})")

    print(f"\nTotal engines registered: {len(_registry.list_engines())}")
    print("Baseplate ready. Bricks are live.")


if __name__ == "__main__":
    main()
