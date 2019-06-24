import numpy as np

def calc_jaccard(x,y, void = None):
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        return 1
    else:
        x = x.astype(np.bool)
        y = y.astype(np.bool)
        if void is None:
            intersection = np.sum((x & y))
            union = np.sum((x | y))
            return float(intersection / union)
        else:
            void = (void > 0.5).astype(np.bool)
            intersection = np.sum(((x & y) & np.logical_not(void)))
            union = np.sum(((x | y) & np.logical_not(void)))
            return float(intersection / union)


def jaccard(annotation, segmentation, void_pixels=None):

    assert(annotation.shape == segmentation.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(annotation)
    assert(void_pixels.shape == annotation.shape)

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(annotation & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(segmentation & np.logical_not(void_pixels)), 0):
        return 1
    else:
        return np.sum(((annotation & segmentation) & np.logical_not(void_pixels))) / \
               np.sum(((annotation | segmentation) & np.logical_not(void_pixels)), dtype=np.float32)

