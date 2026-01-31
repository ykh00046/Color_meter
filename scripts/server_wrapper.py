"""
Server Wrapper - Ensures child process terminates when parent terminates.

Uses Windows Job Object to bind uvicorn process to parent,
so when terminal is closed, server is automatically killed.
"""

import os
import signal
import subprocess
import sys


def main():
    # Windows-specific: Create Job Object to ensure child dies with parent
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Job Object constants
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        # Create job object
        job = kernel32.CreateJobObjectW(None, None)
        if job:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
            )

    # Start uvicorn server
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.web.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        "info",
    ]

    proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)

    # Assign to job object (Windows)
    if sys.platform == "win32" and job:
        kernel32.AssignProcessToJobObject(job, int(proc._handle))

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for process
    try:
        proc.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
