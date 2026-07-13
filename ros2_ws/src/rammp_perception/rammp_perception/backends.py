"""Pluggable detection backends. Each returns, for an RGB image (H,W,3 uint8),
a list of Detection(label, mask, score) — the node turns masks into 3D.

  * ColorBackend — the sim backend: segments the props by their known
    scene.yaml colors. Zero model dependencies, runs anywhere, and exercises
    the entire camera->world pipeline the same way a neural backend will.
  * NanoOWL (Jetson TensorRT, open-vocabulary) is the planned phase-2
    backend for real-world objects — it plugs in here.
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


def make_backend(name, classes, prompts):
    if name == 'color':
        return ColorBackend(classes)
    raise ValueError("unknown backend %r (only 'color' exists; NanoOWL is "
                     'the planned phase-2 backend)' % name)
