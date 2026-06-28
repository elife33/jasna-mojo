from jasna.py_compat import _hasattr
# Restoration pipeline — orchestrates primary and secondary restoration.
# Manages the flow: raw crops → primary model → (optional denoise) → secondary model → output

from std.python import Python, PythonObject
from jasna.pipeline_items import (
    PrimaryRestoreResult,
    SecondaryRestoreResult,
    RestoreResultBase,
    RawCrop,
    TrackedClip,
)
from jasna.crop_buffer import prepare_crops_for_restoration
from jasna.restorer.denoise import (
    DenoiseStrength,
    DenoiseStep,
    apply_denoise,
    apply_denoise_u8,
)
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer


struct RestorationPipeline:
    """Orchestrates primary and secondary restoration steps.
    
    Pipeline flow:
    1. Prepare raw crops (resize, pad to 256x256)
    2. Run primary restoration (BasicVSR++)
    3. Optional denoise after primary
    4. Run secondary restoration (optional upscaling)
    5. Optional denoise after secondary
    """

    var restorer: BasicvsrppMosaicRestorer
    var secondary_restorer: PythonObject  # None or secondary restorer
    var _denoise_strength: DenoiseStrength
    var _denoise_step: DenoiseStep

    def __init__(
        mut self,
        restorer: BasicvsrppMosaicRestorer,
        secondary_restorer: PythonObject = PythonObject(),
        denoise_strength: DenoiseStrength = DenoiseStrength.NONE(),
        denoise_step: DenoiseStep = DenoiseStep.AFTER_PRIMARY(),
    ):
        self.restorer = restorer
        self.secondary_restorer = secondary_restorer
        self._denoise_strength = denoise_strength
        self._denoise_step = denoise_step

        var sec_name = "none"
        if secondary_restorer is not None:
            sec_name = String(py=secondary_restorer.name)
        print("RestorationPipeline: secondary=" + sec_name +
              " denoise=" + denoise_strength.value +
              " denoise_step=" + denoise_step.value)

    def secondary_num_workers(self) raises -> Int:
        """Number of secondary restoration workers."""
        if self.secondary_restorer is not None:
            return Int(py=self.secondary_restorer.num_workers)
        return 1

    def secondary_prefers_cpu_input(self) raises -> Bool:
        """Whether secondary restorer prefers CPU input tensors."""
        if self.secondary_restorer is not None:
            return Bool(getattr(self.secondary_restorer, "prefers_cpu_input", False))
        return False

    def _apply_denoise(self, frames: PythonObject) raises -> PythonObject:
        """Apply denoise to float frames."""
        return apply_denoise(frames, self._denoise_strength)

    def _prepare_from_raw_crops(
        self,
        raw_crops: List[RawCrop],
    ) -> Tuple[
        List[PythonObject],
        List[Tuple[Int, Int, Int, Int]],
        List[Tuple[Int, Int]],
        List[Tuple[Int, Int]],
        List[Tuple[Int, Int]],
    ]:
        """Prepare raw crops for restoration.
        
        Returns:
            (resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes)
        """
        var (resized_crops, pad_offsets, resize_shapes) = prepare_crops_for_restoration(
            raw_crops, self.restorer.device
        )

        var enlarged_bboxes = List[Tuple[Int, Int, Int, Int]]()
        var crop_shapes = List[Tuple[Int, Int]]()
        for c in raw_crops:
            enlarged_bboxes.append((
                c.enlarged_bbox.x1, c.enlarged_bbox.y1,
                c.enlarged_bbox.x2, c.enlarged_bbox.y2,
            ))
            crop_shapes.append((c.crop_h, c.crop_w))

        return (resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes)

    def _run_secondary(
        self,
        primary_raw: PythonObject,
        keep_start: Int,
        keep_end: Int,
    ) raises -> List[PythonObject]:
        """Run secondary restoration on primary output.
        
        Args:
            primary_raw: (T, C, 256, 256) float tensor [0, 1]
            keep_start, keep_end: Frame indices to keep
        Returns:
            List of (C, H, W) uint8 tensors
        """
        var torch = Python.import_module("torch")

        var restored_frames = List[PythonObject]()

        if self.secondary_restorer is not None:
            var result = self.secondary_restorer.restore(
                primary_raw, keep_start=keep_start, keep_end=keep_end
            )
            if Bool(py=_hasattr(result, "dim")):
                # Single tensor — unbind into list
                if Int(result.dim()) > 3:
                    restored_frames = result.unbind(0)
                else:
                    restored_frames.append(result)
            else:
                # Already a list
                restored_frames = result
        else:
            # No secondary — just convert to uint8
            var kept = primary_raw[keep_start:keep_end]
            var processed = kept.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
            restored_frames = processed.unbind(0)

        if self._denoise_step == DenoiseStep.AFTER_SECONDARY():
            var batch_u8 = torch.stack(restored_frames, dim=0)
            var denoised = apply_denoise_u8(batch_u8, self._denoise_strength)
            restored_frames = denoised.unbind(0)

        return restored_frames

    def prepare_and_run_primary(
        self,
        clip: TrackedClip,
        raw_crops: List[RawCrop],
        frame_h: Int,
        frame_w: Int,
        keep_start: Int,
        keep_end: Int,
        crossfade_weights: Dict[Int, Float64],
    ) raises -> PrimaryRestoreResult:
        """Prepare crops and run primary restoration.
        
        Args:
            clip: Tracked clip metadata
            raw_crops: Raw crops for this clip
            frame_h, frame_w: Frame dimensions
            keep_start, keep_end: Frame indices to keep
            crossfade_weights: Weights for crossfade blending
        Returns:
            PrimaryRestoreResult
        """
        var (resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes) =
            self._prepare_from_raw_crops(raw_crops)

        var primary_raw = self.restorer.raw_process(resized_crops)

        if self._denoise_step == DenoiseStep.AFTER_PRIMARY():
            primary_raw = self._apply_denoise(primary_raw)

        var base = RestoreResultBase(
            track_id=clip.track_id,
            start_frame=clip.start_frame,
            frame_count=len(raw_crops),
            frame_h=frame_h,
            frame_w=frame_w,
            frame_device=raw_crops[0].crop.device,
        )
        base.keep_start = keep_start
        base.keep_end = keep_end
        base.crossfade_weights = crossfade_weights

        # Convert bboxes and shapes to lists
        for i in range(len(clip.masks)):
            base.masks.append(clip.masks[i])

        for i in range(len(enlarged_bboxes)):
            base.enlarged_bboxes.append(enlarged_bboxes[i])
            base.crop_shapes.append(crop_shapes[i])
            base.pad_offsets.append(pad_offsets[i])
            base.resize_shapes.append(resize_shapes[i])

        return PrimaryRestoreResult(base, primary_raw)

    def build_secondary_result(
        self,
        pr: PrimaryRestoreResult,
        restored_frames: List[PythonObject],
    ) raises -> SecondaryRestoreResult:
        """Build secondary result from primary result and restored frames.
        
        Args:
            pr: Primary restoration result
            restored_frames: List of restored frame tensors
        Returns:
            SecondaryRestoreResult
        """
        var torch = Python.import_module("torch")

        var frames = restored_frames
        if self._denoise_step == DenoiseStep.AFTER_SECONDARY():
            var batch_u8 = torch.stack(frames, dim=0)
            var denoised = apply_denoise_u8(batch_u8, self._denoise_strength)
            frames = denoised.unbind(0)

        var ks = max(0, pr.base.keep_start)
        var ke = min(pr.base.frame_count, pr.base.keep_end)
        var kept_count = ke - ks

        var base = RestoreResultBase(
            track_id=pr.base.track_id,
            start_frame=pr.base.start_frame,
            frame_count=pr.base.frame_count,
            frame_h=pr.base.frame_h,
            frame_w=pr.base.frame_w,
            frame_device=pr.base.frame_device,
        )
        base.keep_start = 0
        base.keep_end = kept_count
        base.crossfade_weights = pr.base.crossfade_weights

        # Slice masks, bboxes, shapes to kept range
        for i in range(ks, ke):
            base.masks.append(pr.base.masks[i])
            base.enlarged_bboxes.append(pr.base.enlarged_bboxes[i])
            base.crop_shapes.append(pr.base.crop_shapes[i])
            base.pad_offsets.append(pr.base.pad_offsets[i])
            base.resize_shapes.append(pr.base.resize_shapes[i])

        return SecondaryRestoreResult(base, frames, ks)
