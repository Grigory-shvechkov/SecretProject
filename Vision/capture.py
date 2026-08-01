import platform
import threading

import cv2

# V4L2 is the backend the Pi needs (see run.py's comment on /dev/video
# nodes), but it's Linux-only -- forcing it elsewhere makes VideoCapture
# fail to open every time, even with a working webcam. Only force it
# where it actually applies; elsewhere let OpenCV pick its platform
# default (DirectShow/MSMF on Windows, AVFoundation on macOS) -- useful
# since this project is edited on Windows but run on the Pi.
_BACKEND = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY

# How long to wait for cv2.VideoCapture(...) to open before giving up on
# it -- guards against a hung open() call the same way the background
# reader thread below guards against a hung read() call.
OPEN_TIMEOUT_SECONDS = 5.0


class Camera:
    """cv2.VideoCapture wrapper that can never block the caller.

    On the Pi, each USB camera exposes two /dev/video nodes and only one
    of the pair actually streams video -- the other is a metadata-only
    node with no formats (see run.py). Indices can also shift across
    reboots/replugs. If FRONT_CAMERA_INDEX/SIDE_CAMERA_INDEX end up
    pointing at the wrong node, cv2.VideoCapture.open() reports success
    (isOpened() is True) but cap.read() then blocks forever waiting for a
    frame that will never arrive -- which freezes the ENTIRE main loop,
    including the cv2.waitKey() call that reads keyboard input, making
    the whole app look hung with no way to quit.

    Fix: a background thread does the actual (possibly-blocking)
    cap.read() calls in a tight loop and stores only the latest result.
    read() here never touches the camera directly -- it just returns
    whatever the background thread most recently captured, so a stalled
    camera degrades to "frames stop updating" instead of "the whole
    program freezes."
    """

    def __init__(self, index=0, width=640, height=480):
        self.index = index
        self.cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = None

        self.cap = self._open_with_timeout(index, OPEN_TIMEOUT_SECONDS)
        if self.cap is None:
            # Don't crash the whole pipeline over one missing/unplugged/
            # hung camera -- print why and leave self.cap as None so
            # read() just reports "no frame" forever, same as a dropped
            # frame.
            print(f"Warning: Camera {index} could not be opened (or hung "
                  f"opening past {OPEN_TIMEOUT_SECONDS:.0f}s). Continuing "
                  f"without it -- frames from it will be None.")
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_with_timeout(self, index, timeout):
        """Run cv2.VideoCapture(...) on a daemon thread and join with a
        timeout, so a hung open() can't hang __init__ (and therefore the
        whole app) -- same rationale as the reader thread. If it never
        finishes, the thread is simply abandoned (daemon=True keeps it
        from blocking process exit); we treat this camera as unavailable
        rather than wait indefinitely.
        """
        result = {}

        def _open():
            result["cap"] = cv2.VideoCapture(index, _BACKEND)

        opener = threading.Thread(target=_open, daemon=True)
        opener.start()
        opener.join(timeout)

        cap = result.get("cap")
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        return cap

    def _reader_loop(self):
        # Only ever started when self._open_with_timeout succeeded, so
        # self.cap is a real VideoCapture here -- pinned to a local so the
        # type checker doesn't have to reason about the Optional field
        # across the thread boundary.
        cap = self.cap
        if cap is None:
            return
        while not self._stopped:
            ok, frame = cap.read()
            with self._lock:
                self._frame = frame if ok else None

    def read(self):
        if self.cap is None:
            return None
        with self._lock:
            return self._frame

    def release(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    #these dunders are for the with x, as y blocks (safe clean up)

    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.release()
