# Python compatibility shim for jasna-mojo.
# This module provides Python-importable wrappers that delegate to the Mojo
# pipeline components, allowing the existing Python ML/video modules to be
# used seamlessly from Mojo via interop.

# When running via `mojo run`, this file sets up the Python path so that
# the original jasna Python package (or its mock equivalents) can be imported.

from std.python import Python, PythonObject


def _hasattr(obj: PythonObject, name: String) raises -> Bool:
    """Python hasattr() wrapper for use from Mojo."""
    var builtins = Python.import_module("builtins")
    return Bool(py=builtins.hasattr(obj, name))


def setup_python_path() raises:
    """Configure Python import paths without trusting the process cwd."""
    var configure = Python.evaluate("""
def _configure_python_path():
    import os
    import sys
    from pathlib import Path

    module = sys.modules.get("jasna.py_compat")
    module_file = getattr(module, "__file__", "") if module is not None else ""
    if module_file:
        repo_root = Path(module_file).resolve().parent.parent
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.append(repo_root_str)

    extra_paths = os.environ.get("JASNA_PYTHON_PATH", "")
    for part in extra_paths.split(os.pathsep):
        if part:
            path = str(Path(part).expanduser().resolve())
            if path not in sys.path:
                sys.path.append(path)

    original_path = os.environ.get("JASNA_ORIGINAL_JASNA_PATH", "")
    if original_path:
        path = Path(original_path).expanduser().resolve()
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)
""", file=True)
    configure._configure_python_path()


def ensure_mock_modules() raises:
    """Ensure mock detection/restoration modules exist for testing without model weights."""
    var create_mocks = Python.evaluate("""
def _ensure_mocks():
    import sys
    import types
    import torch
    import numpy as np

    # Mock detection model
    if 'jasna.mock_detection' not in sys.modules:
        mock_det = types.ModuleType('jasna.mock_detection')
        
        class MockDetectionModel:
            def __init__(self, device):
                self.device = device
                self.name = "mock"
            def __call__(self, frames, target_hw=None):
                from jasna.mosaic.detections import Detections
                B = frames.shape[0]
                boxes_list = []
                masks_list = []
                for i in range(B):
                    boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                    h, w = target_hw if target_hw else (1080, 1920)
                    masks_list.append(torch.zeros((0, h//4, w//4), dtype=torch.bool))
                return Detections(boxes_xyxy=boxes_list, masks=masks_list)
            def close(self):
                pass
        
        def create_mock_detection_model(device):
            return MockDetectionModel(device)
        
        mock_det.MockDetectionModel = MockDetectionModel
        mock_det.create_mock_detection_model = create_mock_detection_model
        sys.modules['jasna.mock_detection'] = mock_det

    # Mock restoration model
    if 'jasna.mock_restoration' not in sys.modules:
        mock_rest = types.ModuleType('jasna.mock_restoration')
        
        class MockBasicvsrppMosaicRestorer:
            def __init__(self, checkpoint_path, device, max_clip_size, use_tensorrt, fp16):
                self.device = device
                self.max_clip_size = max_clip_size
                self.dtype = torch.float16 if fp16 else torch.float32
            def close(self):
                pass
            def raw_process(self, video):
                with torch.inference_mode():
                    T = len(video)
                    return torch.zeros((T, 3, 256, 256), device=self.device, dtype=self.dtype)
            def restore(self, video):
                result = self.raw_process(video)
                result = result.mul(255).round().clamp(0,255).to(torch.uint8).permute(0,2,3,1)
                return list(torch.unbind(result, 0))
        
        mock_rest.MockBasicvsrppMosaicRestorer = MockBasicvsrppMosaicRestorer
        sys.modules['jasna.mock_restoration'] = mock_rest

    # Mock bootstrap
    if 'jasna.bootstrap' not in sys.modules:
        mock_boot = types.ModuleType('jasna.bootstrap')
        def sanitize_sys_path_for_local_dev(path):
            pass
        mock_boot.sanitize_sys_path_for_local_dev = sanitize_sys_path_for_local_dev
        sys.modules['jasna.bootstrap'] = mock_boot

    # Mock engine compiler
    if 'jasna.engine_compiler' not in sys.modules:
        mock_ec = types.ModuleType('jasna.engine_compiler')
        from dataclasses import dataclass
        from typing import Optional
        
        @dataclass
        class EngineCompilationRequest:
            device: str = "cpu"
            fp16: bool = False
            basicvsrpp: bool = False
            basicvsrpp_model_path: str = ""
            basicvsrpp_max_clip_size: int = 90
            detection: bool = False
            detection_model_name: str = "rfdetr-v5"
            detection_model_path: str = ""
            detection_batch_size: int = 4
            unet4x: bool = False
            
            def to_json(self):
                import json
                return json.dumps({
                    'device': self.device, 'fp16': self.fp16,
                    'basicvsrpp': self.basicvsrpp,
                    'basicvsrpp_model_path': self.basicvsrpp_model_path,
                    'basicvsrpp_max_clip_size': self.basicvsrpp_max_clip_size,
                    'detection': self.detection,
                    'detection_model_name': self.detection_model_name,
                    'detection_model_path': self.detection_model_path,
                    'detection_batch_size': self.detection_batch_size,
                    'unet4x': self.unet4x,
                })
            
            @staticmethod
            def from_json(s):
                import json
                d = json.loads(s)
                return EngineCompilationRequest(**d)
        
        class _MockResult:
            use_basicvsrpp_tensorrt = False
        
        def ensure_engines_compiled(req):
            return _MockResult()
        
        def _subprocess_compile(req):
            pass
        
        mock_ec.EngineCompilationRequest = EngineCompilationRequest
        mock_ec.ensure_engines_compiled = ensure_engines_compiled
        mock_ec._subprocess_compile = _subprocess_compile
        sys.modules['jasna.engine_compiler'] = mock_ec

    # Mock _suppress_noise
    if 'jasna._suppress_noise' not in sys.modules:
        mock_sn = types.ModuleType('jasna._suppress_noise')
        def install():
            pass
        mock_sn.install = install
        sys.modules['jasna._suppress_noise'] = mock_sn

    # Mock benchmark
    if 'jasna.benchmark' not in sys.modules:
        mock_bm = types.ModuleType('jasna.benchmark')
        def run_benchmark_cli(args):
            print("Benchmark mode not available in Mojo port")
        mock_bm.run_benchmark_cli = run_benchmark_cli
        sys.modules['jasna.benchmark'] = mock_bm

    # Mock mmengine only if not installed
    try:
        import mmengine
    except ImportError:
        import types as _types
        import sys as _sys
        
        class _AutoMock:
            def __init__(self, *a, **k):
                name = a[0] if a else k.get('name', 'mock')
                object.__setattr__(self, '_name', name)
                object.__setattr__(self, '__path__', [])
                object.__setattr__(self, '__package__', name)
                object.__setattr__(self, '__spec__', None)
                object.__setattr__(self, '__loader__', None)
                object.__setattr__(self, '__file__', None)
            def __getattr__(self, name):
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                if name.startswith('_'):
                    raise AttributeError(name)
                obj = _AutoMock(self._name + '.' + name)
                object.__setattr__(self, name, obj)
                return obj
            def __call__(self, *a, **k):
                return _AutoMock(self._name + '()')
        
        def _make_mock_module(modname):
            mod = _AutoMock(modname)
            _sys.modules[modname] = mod
            return mod
        
        # Install mmengine with auto-mock for any submodule
        mock_mme = _make_mock_module('mmengine')
        
        # Meta path finder to auto-mock any mmengine.* import
        from importlib.abc import MetaPathFinder
        from importlib.machinery import ModuleSpec
        class _AutoMockLoader:
            def create_module(self, spec):
                return _AutoMock(spec.name)
            def exec_module(self, module):
                pass
        class _MmengineFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                prefixes = ('mmengine.', 'mmcv.', 'mmdet.', 'psutil', 'psutil.',
                            'mmengine', 'mmcv', 'mmdet')
                if any(fullname == p or fullname.startswith(p + '.') for p in prefixes):
                    return ModuleSpec(fullname, _AutoMockLoader(), is_package=True)
                return None
        
        _sys.meta_path.insert(0, _MmengineFinder())

    # Mock mmcv only if not installed
    try:
        import mmcv
    except ImportError:
        if 'mmcv' not in sys.modules:
            mock_mmcv = types.ModuleType('mmcv')
            mock_mmcv.__path__ = []
            sys.modules['mmcv'] = mock_mmcv

    # Mock torchvision only if not installed
    try:
        import torchvision
    except ImportError:
        if 'torchvision' not in sys.modules:
            mock_tvs = types.ModuleType('torchvision')
            mock_tvs.__path__ = []
            mock_tvs_models = types.ModuleType('torchvision.models')
            mock_tvs.models = mock_tvs_models
            sys.modules['torchvision'] = mock_tvs
            sys.modules['torchvision.models'] = mock_tvs_models
""", file=True)
    create_mocks._ensure_mocks()
