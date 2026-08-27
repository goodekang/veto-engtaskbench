from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from .blackboard import SubGoal
from .data import TaskCache


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float = 120.0
    memory_mb: int = 2048
    max_output_mb: int = 32

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("sandbox timeout must be positive")
        if self.memory_mb <= 0 or self.max_output_mb <= 0:
            raise ValueError("sandbox memory and output limits must be positive")


def _apply_posix_limits(limits: SandboxLimits) -> None:
    if os.name == "nt":
        return
    import resource

    memory = limits.memory_mb * 1024 * 1024
    output = limits.max_output_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (max(1, int(limits.timeout_s)), max(2, int(limits.timeout_s) + 1)),
    )


class _WindowsJob(AbstractContextManager):
    """Assign a spawned tool worker to a memory-capped Windows Job Object."""

    def __init__(self, pid: int, limits: SandboxLimits):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMITS(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMITS),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = EXTENDED_LIMITS()
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
        info.ProcessMemoryLimit = limits.memory_mb * 1024 * 1024
        if not kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)
        if not process:
            kernel32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            if not kernel32.AssignProcessToJobObject(handle, process):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        except Exception:
            kernel32.CloseHandle(handle)
            raise
        finally:
            kernel32.CloseHandle(process)
        self.handle = handle
        self._kernel32 = kernel32

    def __exit__(self, exc_type, exc_value, traceback_value):
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None
        return False


def _worker(
    out: mp.Queue,
    tool: dict[str, Any],
    args: dict[str, Any],
    cache: TaskCache,
    goal: SubGoal,
    limits: SandboxLimits,
) -> None:
    try:
        _apply_posix_limits(limits)
        from .executor import run_tool

        result = run_tool(tool, args, cache, goal, board=None)
        out.put(("ok", result))
    except BaseException as exc:  # process boundary must return a typed failure
        out.put(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )
        )


def execute_sandboxed(
    tool: dict[str, Any],
    args: dict[str, Any],
    cache: TaskCache,
    goal: SubGoal,
    limits: SandboxLimits | None = None,
) -> dict[str, Any]:
    """Execute one cached tool call behind a process and timeout boundary."""
    limits = limits or SandboxLimits()
    context = mp.get_context("spawn")
    out: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker,
        args=(out, tool, args, cache, goal, limits),
        daemon=True,
    )
    process.start()
    try:
        job = _WindowsJob(process.pid, limits)
    except Exception:
        process.terminate()
        process.join(5)
        out.close()
        out.join_thread()
        raise
    with job:
        process.join(limits.timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(5)
            return {
                "ok": False,
                "tool": tool.get("name"),
                "error": "timeout",
                "details": [f"tool exceeded {limits.timeout_s:.1f}s"],
            }
        try:
            status, payload = out.get_nowait()
        except queue.Empty:
            return {
                "ok": False,
                "tool": tool.get("name"),
                "error": "sandbox_exit",
                "details": [f"worker exited with code {process.exitcode} without a result"],
            }
        finally:
            out.close()
            out.join_thread()
    if status == "error":
        return {
            "ok": False,
            "tool": tool.get("name"),
            "error": "sandbox_exception",
            "details": [payload["type"], payload["message"]],
            "worker_traceback": payload["traceback"],
        }
    return payload
