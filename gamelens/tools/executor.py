"""Trusted registered tools run in killable processes, never detached threads."""
import ctypes
import multiprocessing
import os
import signal
import time

import cloudpickle

from gamelens.domain.tool_results import ToolFailure, classify_tool_exception


class ProcessJob:
    """Windows kills assigned descendants when the owning Agent exits."""
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit",ctypes.c_int64),("PerJobUserTimeLimit",ctypes.c_int64),
                ("LimitFlags",w.DWORD),("MinimumWorkingSetSize",ctypes.c_size_t),("MaximumWorkingSetSize",ctypes.c_size_t),
                ("ActiveProcessLimit",w.DWORD),("Affinity",ctypes.c_size_t),("PriorityClass",w.DWORD),("SchedulingClass",w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(key,ctypes.c_uint64) for key in ("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]
        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation",Basic),("IoInfo",IO),("ProcessMemoryLimit",ctypes.c_size_t),
                ("JobMemoryLimit",ctypes.c_size_t),("PeakProcessMemoryUsed",ctypes.c_size_t),("PeakJobMemoryUsed",ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p,w.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.OpenProcess.argtypes = [w.DWORD,w.BOOL,w.DWORD]
        self.handle = self.kernel.CreateJobObjectW(None,None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self,pid):
        if self.handle is None:
            return
        handle = self.kernel.OpenProcess(0x0100 | 0x0001,False,pid)
        try:
            if not handle or not self.kernel.AssignProcessToJobObject(self.handle,handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if handle: self.kernel.CloseHandle(handle)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _worker(payload, sender, start_event, parent_pid):
    if os.name == "posix":
        # Linux parent-death signal; an abruptly terminated Agent leaves no worker.
        try:
            ctypes.CDLL(None).prctl(1,signal.SIGKILL)
            if os.getppid() != parent_pid:
                return
        except (AttributeError,OSError):
            pass
    start_event.wait()
    try:
        handler,arguments,state = cloudpickle.loads(payload)
        result = handler(arguments,state)
        sender.send_bytes(cloudpickle.dumps(("result",result.model_dump(mode="json"))))
    except BaseException as error:
        failure = classify_tool_exception(error)
        sender.send_bytes(cloudpickle.dumps(("error", failure.model_dump(mode="json"))))
    finally:
        sender.close()


def run_isolated(handler,arguments,state,timeout):
    payload=cloudpickle.dumps((handler,arguments,state))
    context = multiprocessing.get_context("spawn")
    receiver,sender = context.Pipe(duplex=False)
    event = context.Event()
    worker = context.Process(target=_worker,args=(payload,sender,event,os.getpid()))
    job = ProcessJob()
    started = time.monotonic()
    try:
        worker.start()
        sender.close()
        job.assign(worker.pid)
        event.set()  # Never execute a tool before its lifecycle protection is attached.
        if not receiver.poll(max(0,timeout-(time.monotonic()-started))):
            failure = ToolFailure(error_type="TimeoutError", error_code="tool_timeout",
                message=f"工具超过{timeout:g}秒", retryable=False,
                side_effects_started=True, status_code=408)
            return "timeout", failure.model_dump(mode="json")
        try:
            return cloudpickle.loads(receiver.recv_bytes())
        except EOFError:
            worker.join(0.2)
            failure = ToolFailure(error_type="WorkerProcessError", error_code="worker_process_died",
                message=f"工具子进程异常退出：{worker.exitcode}", retryable=False,
                side_effects_started=True, status_code=None)
            return "error", failure.model_dump(mode="json")
    finally:
        if worker.pid is not None:
            if worker.is_alive(): worker.terminate()
            worker.join(2)
            if worker.is_alive(): worker.kill(); worker.join(2)
        job.close()
        receiver.close()
        sender.close()
