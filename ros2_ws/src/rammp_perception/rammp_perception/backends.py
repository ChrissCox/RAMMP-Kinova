"""Pluggable detection backends. Each returns, for an RGB image (H,W,3 uint8),
a list of Detection(label, mask, score) — the node turns masks into 3D.

  * ColorBackend — sim bring-up backend: segments the props by their known
    scene.yaml colors. Zero model dependencies, runs anywhere, validates the
    entire camera->world pipeline before any neural network is installed.
  * OwlVitBackend — real open-vocabulary detection via HuggingFace
    transformers (OWL-ViT / OWLv2). Works on any torch install (CPU or GPU);
    the stepping stone to NanoOWL.
  * NanoOWL (Jetson TensorRT) is the production target: it emits
    vision_msgs/Detection2DArray from its own ROS node — when it is running,
    point the detector node's `external_detections_topic` at it instead of
    using an in-process backend.
"""

import numpy as np


class Detection:
    __slots__ = ('label', 'mask', 'score')

    def __init__(self, label, mask, score):
        self.label = label
        self.mask = mask          # bool (H, W)
        self.score = float(score)


class ColorBackend:
    """Nearest-known-color segmentation for the sim props.

    classes: {label: (r, g, b) in 0..1}. Pixels within `tol` (normalized RGB
    distance) of a class color and part of a big-enough blob become that
    label's mask. Labels sharing near-identical colors (mug/apple are both
    red) yield MULTIPLE detections with the same color class — the node
    disambiguates by position continuity.
    """

    def __init__(self, classes, tol=0.08, min_pixels=40):
        # Chromatic classes only: white/cream props (plate, bowl) are
        # indistinguishable from the white arm and beige wall by color —
        # they keep their YAML poses until a neural backend tracks them.
        # Matching is by CHROMATICITY (rgb / sum): lighting and specular
        # highlights scale brightness, not color ratios (absolute-RGB
        # matching lost the mug to its own highlights).
        self.classes = {k: np.asarray(v, float) for k, v in classes.items()
                        if max(v) - min(v) >= 0.2}
        self.tol = float(tol)
        self.min_pixels = int(min_pixels)

    @staticmethod
    def _chroma(img):
        s = img.sum(axis=-1, keepdims=True)
        return img / np.maximum(s, 1e-6)

    def detect(self, rgb):
        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None
        img = rgb.astype(float) / 255.0
        chroma = self._chroma(img)
        bright = img.sum(axis=2) > 0.25   # skip shadows/black
        out = []
        for label, color in self.classes.items():
            cc = self._chroma(color[None, None, :])[0, 0]
            dist = np.linalg.norm(chroma - cc[None, None, :], axis=2)
            m = (dist < self.tol) & bright
            if m.sum() < self.min_pixels:
                continue
            if ndimage is not None:
                lab, n = ndimage.label(m)
                for i in range(1, n + 1):
                    blob = lab == i
                    if blob.sum() >= self.min_pixels:
                        out.append(Detection(label, blob, 0.9))
            else:
                out.append(Detection(label, m, 0.7))
        return out


class OwlVitBackend:
    """HuggingFace OWL-ViT: text-prompted open-vocab boxes -> box masks.

    pip install transformers torch pillow. First call downloads the model
    (~600 MB). Slow on CPU (~seconds/frame) — fine at low rates; GPU-capable
    wherever torch.cuda is available.
    """

    def __init__(self, prompts, threshold=0.15,
                 model_id='google/owlvit-base-patch32'):
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.prompts = list(prompts)
        self.threshold = float(threshold)

    def detect(self, rgb):
        from PIL import Image
        torch = self.torch
        image = Image.fromarray(rgb)
        texts = [['a photo of a %s' % p for p in self.prompts]]
        inputs = self.processor(text=texts, images=image,
                                return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target = torch.tensor([rgb.shape[:2]], device=self.device)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target)[0]
        out = []
        h, w = rgb.shape[:2]
        for score, lab, box in zip(res['scores'], res['labels'], res['boxes']):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = True
            out.append(Detection(self.prompts[int(lab)], mask, float(score)))
        return out


def make_backend(name, classes, prompts):
    if name == 'color':
        return ColorBackend(classes)
    if name == 'owlvit':
        return OwlVitBackend(prompts)
    raise ValueError('unknown backend %r (color | owlvit)' % name)
