# BasicVSR++ mosaic restorer — loads and runs the restoration model.
# Uses Python interop for PyTorch model loading and inference.

from std.python import Python, PythonObject


comptime INFERENCE_SIZE = 256


struct BasicvsrppMosaicRestorer:
    """BasicVSR++ mosaic restoration model wrapper.
    
    Loads the model from a checkpoint and provides raw_process/restore methods.
    Optionally uses TensorRT sub-engines for faster inference on NVIDIA GPUs.
    """

    var device: PythonObject
    var max_clip_size: Int
    var use_tensorrt: Bool
    var dtype: PythonObject
    var input_dtype: PythonObject
    var _split_forward: PythonObject  # None or split forward callable
    var model: PythonObject  # None or PyTorch model

    def __init__(
        mut self,
        checkpoint_path: String,
        device: PythonObject,
        max_clip_size: Int,
        use_tensorrt: Bool,
        fp16: Bool,
        config: PythonObject = PythonObject(),
    ):
        var torch = Python.import_module("torch")

        self.device = torch.device(device)
        self.max_clip_size = max_clip_size
        self.use_tensorrt = use_tensorrt
        self.dtype = torch.float16 if fp16 else torch.float32
        self.input_dtype = self.dtype
        self._split_forward = PythonObject()
        self.model = PythonObject()

        if use_tensorrt and String(py=self.device.type) == "cuda":
            # Try to load with TensorRT sub-engines
            var load_model = Python.evaluate("""
def _load_model(config, checkpoint_path, device, fp16):
    from jasna.models.basicvsrpp.inference import load_model
    return load_model._load_model(config, checkpoint_path, device, fp16)
""", file=True)

            var pytorch_model = load_model._load_model(config, checkpoint_path, self.device, fp16)

            # Try to create split forward (TensorRT)
            var create_split = Python.evaluate("""
def _try_create_split(model, weights_path, device, fp16, max_clip_size):
    try:
        from jasna.restorer.basicvsrpp_sub_engines import create_split_forward
        return create_split_forward(
            model=model,
            model_weights_path=weights_path,
            device=device,
            fp16=fp16,
            max_clip_size=max_clip_size,
        )
    except Exception:
        return None
""", file=True)

            self._split_forward = create_split._try_create_split(
                pytorch_model, checkpoint_path, self.device, fp16, max_clip_size
            )

            if self._split_forward is not None:
                print("BasicVSR++ using TRT sub-engines (fp16=" + String(fp16) + ")")
            else:
                self.model = pytorch_model
                print("BasicVSR++ sub-engines not found, using PyTorch model (fp16=" + String(fp16) + ")")
        else:
            var load_model = Python.evaluate("""
def _load_model(config, checkpoint_path, device, fp16):
    from jasna.models.basicvsrpp.inference import load_model
    return load_model._load_model(config, checkpoint_path, device, fp16)
""", file=True)

            self.model = load_model._load_model(config, checkpoint_path, self.device, fp16)
            print("BasicVSR++ loaded from checkpoint: " + checkpoint_path + " (fp16=" + String(fp16) + ")")

    def close(mut self) raises:
        """Close and release model resources."""
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = PythonObject()
        self.model = PythonObject()

    def raw_process(self, video: List[PythonObject]) raises -> PythonObject:
        """Process video frames through the restoration model.
        
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        var torch = Python.import_module("torch")

        with torch.inference_mode():
            var stacked = torch.stack(video).permute(0, 3, 1, 2).to(
                device=self.device,
                dtype=self.input_dtype,
                memory_format=torch.contiguous_format,
            ).div_(255.0)

            if self._split_forward is not None:
                var result = self._split_forward(stacked.unsqueeze(0))
                return result.squeeze(0)
            else:
                var result = self.model(inputs=stacked.unsqueeze(0))
                return result.squeeze(0)

    def restore(self, video: List[PythonObject]) raises -> List[PythonObject]:
        """Restore video frames and return uint8 tensors.
        
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            list of (256, 256, C) uint8 tensors in RGB format
        """
        var torch = Python.import_module("torch")
        var result = self.raw_process(video)
        var processed = result.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1)
        var frames_list = torch.unbind(processed, 0)
        return frames_list
