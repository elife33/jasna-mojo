# Progress bar with time remaining estimation and speed display.
# Uses Python interop for tqdm.

from std.python import Python, PythonObject


struct Progressbar:
    """Progress bar with time remaining estimation and speed display."""

    var total_frames: Int
    var callback: PythonObject  # Optional callback
    var frames_processed: Int
    var _durations_buffer: List[Float64]
    var _buffer_min_len: Int
    var _buffer_max_len: Int
    var error: Bool
    var disable: Bool
    var _tqdm: PythonObject
    var _duration_start: Float64

    def __init__(
        mut self,
        total_frames: Int,
        video_fps: Float64,
        disable: Bool = False,
        callback: PythonObject = PythonObject(),
    ):
        self.total_frames = total_frames
        self.callback = callback
        self.frames_processed = 0
        self._durations_buffer = List[Float64]()
        self._buffer_min_len = min(total_frames - 1, Int(video_fps * 2.0))
        self._buffer_max_len = min(total_frames - 1, Int(video_fps * 30.0))
        self.error = False
        self.disable = disable
        self._duration_start = 0.0

        var tqdm = Python.import_module("tqdm")

        var bar_format = (
            "Processing video: {percentage:3.0f}%|{bar}|"
            "Processed: {elapsed} ({n_fmt}f){desc}"
        )
        var initial_suffix = " | Remaining: ? | Speed: ?"

        self._tqdm = tqdm.tqdm(
            dynamic_ncols=True,
            total=total_frames,
            bar_format=bar_format,
            desc=initial_suffix,
            disable=disable,
        )

    def init(self) raises:
        """Initialize the timer."""
        var time = Python.import_module("time")
        self._duration_start = Float64(time.time())

    def close(self, ensure_completed_bar: Bool = False) raises:
        """Close the progress bar."""
        if ensure_completed_bar:
            if not self.error and Int(py=self._tqdm.total) != Int(py=self._tqdm.n):
                self._tqdm.total = self._tqdm.n
                self._update_time_remaining_and_speed(completed=True)
                self._tqdm.refresh()
        self._tqdm.close()

    def update(self, n: Int = 1) raises:
        """Update progress after processing n frames."""
        var time = Python.import_module("time")

        if self._duration_start == 0.0:
            self.init()

        self.frames_processed += n

        var now = Float64(time.time())
        var duration = now - self._duration_start
        self._duration_start = now

        var per_frame = duration / Float64(n) if n > 0 else duration
        for _ in range(n):
            if len(self._durations_buffer) >= self._buffer_max_len:
                self._durations_buffer.pop(0)
            self._durations_buffer.append(per_frame)

        self._update_time_remaining_and_speed()
        self._tqdm.update(n)

        if self.callback is not None:
            var progress_pct = (Float64(py=self.frames_processed) / Float64(py=self.total_frames)) * 100.0 if self.total_frames > 0 else 0.0
            var fps = 0.0
            var eta = 0.0
            if len(self._durations_buffer) > self._buffer_min_len:
                var mean_dur = self._get_mean_duration()
                fps = 1.0 / mean_dur if mean_dur > 0 else 0.0
                var remaining = self.total_frames - self.frames_processed
                eta = Float64(remaining) * mean_dur
            self.callback(progress_pct, fps, eta, self.frames_processed, self.total_frames)

    def _get_mean_duration(self) raises -> Float64:
        """Calculate mean frame processing duration."""
        var total = 0.0
        for d in self._durations_buffer:
            total += d
        return total / Float64(len(self._durations_buffer))

    def _format_duration(self, duration_s: Float64) raises -> String:
        """Format duration in seconds to human-readable string."""
        if duration_s <= 0:
            return "0:00"
        var seconds = Int(duration_s)
        var minutes = seconds // 60
        var hours = minutes // 60
        var s = seconds % 60
        var m = minutes % 60
        if hours == 0:
            return String(m) + ":" + ("0" if s < 10 else "") + String(s)
        return String(hours) + ":" + ("0" if m < 10 else "") + String(m) + ":" + ("0" if s < 10 else "") + String(s)

    def _update_time_remaining_and_speed(self, completed: Bool = False) raises:
        """Update the description with remaining time and processing speed."""
        var total = Int(py=self._tqdm.format_dict["total"])
        var n = Int(py=self._tqdm.format_dict["n"])
        var frames_remaining = 0 if completed else total - n

        if len(self._durations_buffer) > self._buffer_min_len:
            var mean_dur = self._get_mean_duration()
            var time_remaining = Float64(frames_remaining) * mean_dur
            var time_str = self._format_duration(time_remaining)
            var speed = "?.?" if mean_dur <= 0 else String(1.0 / mean_dur)
            # Limit to 1 decimal
            var speed_fps = speed[:speed.find(".") + 2] if speed.find(".") >= 0 else speed
            self._tqdm.desc = " | Remaining: " + time_str + " (" + String(frames_remaining) + "f) | Speed: " + speed_fps + "fps"
