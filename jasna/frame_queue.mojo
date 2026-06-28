# Thread-safe frame queue with frame-count-based backpressure.
# Native Mojo implementation using Python's threading primitives for
# compatibility with Python-based pipeline threads.

from std.python import Python, PythonObject


struct FrameQueue:
    """A thread-safe queue with frame-count-based backpressure.
    
    Items are arbitrary Python objects. Each item has an associated frame_count
    that contributes to the total frames in the queue. When the total exceeds
    max_frames, put() blocks until space is available.
    """

    var _py_queue: PythonObject  # Python object wrapping our custom queue
    var _max_frames: Int

    def __init__(out self, max_frames: Int) raises:
        self._max_frames = max_frames
        var threading = Python.import_module("threading")
        var collections = Python.import_module("collections")

        # Create a Python-based queue with condition variable
        self._py_queue = Python.evaluate("""
import threading
from collections import deque

class _FrameQueue:
    def __init__(self, max_frames) raises:
        self._deque = deque()
        self._cond = threading.Condition()
        self._max_frames = max_frames
        self._current_frames = 0
        self._unfinished_tasks = 0

    def put(self, item, frame_count=0) raises:
        with self._cond:
            if frame_count > 0:
                while self._current_frames > 0 and self._current_frames + frame_count > self._max_frames:
                    self._cond.wait()
            self._deque.append((item, frame_count))
            self._current_frames += frame_count
            self._unfinished_tasks += 1
            self._cond.notify_all()

    def get(self, timeout=None) raises:
        with self._cond:
            if not self._deque:
                self._cond.wait(timeout=timeout)
            if not self._deque:
                raise Empty
            item, frame_count = self._deque.popleft()
            self._current_frames -= frame_count
            self._cond.notify_all()
            return item

    def get_nowait(self) raises:
        with self._cond:
            if not self._deque:
                raise Empty
            item, frame_count = self._deque.popleft()
            self._current_frames -= frame_count
            self._cond.notify_all()
            return item

    def qsize(self) raises:
        with self._cond:
            return len(self._deque)

    def empty(self) raises:
        with self._cond:
            return len(self._deque) == 0

    def current_frames(self) raises:
        with self._cond:
            return self._current_frames
""")
        self._py_queue = self._py_queue(max_frames)

    def put(self, item: PythonObject, frame_count: Int = 0) raises:
        self._py_queue.put(item, frame_count)

    def get(self, timeout: Float64 = -1.0) raises -> PythonObject:
        if timeout < 0:
            return self._py_queue.get()
        else:
            return self._py_queue.get(timeout=timeout)

    def get_nowait(self) raises -> PythonObject:
        return self._py_queue.get_nowait()

    def qsize(self) raises -> Int:
        return Int(py=self._py_queue.qsize())

    def empty(self) raises -> Bool:
        return Bool(py=self._py_queue.empty())

    def current_frames(self) raises -> Int:
        return Int(py=self._py_queue.current_frames())
