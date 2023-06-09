import numpy as np

class image_wrapper:
    def __init__(self, input_img, segments, preprocessor):
        self.input = input_img
        self.input_np = np.asarray(input_img)
        self.segments = segments
        self.num_features = len(np.unique(segments))
        self.preprocessor = preprocessor

    def _preprocess(self, img):
        batch = self.preprocessor(img).unsqueeze(0)
        return batch

    def _get_mask_indices_of_feature(self, feature):
        if isinstance(feature, int):
            feature = [feature]
            return np.argwhere(self.segments == feature)
        else:
            idx_l = []
            for feature_idx in feature:
                idx_l.append(np.argwhere(self.segments == feature_idx))
            return idx_l

    def _transform(self, mask_indices, m):
        img = self.input_np.copy()
        if mask_indices.ndim == 1 or mask_indices.ndim == 3:
            for i in range(len(mask_indices)):
                img[mask_indices[i][:, 0], mask_indices[i][:, 1], :] = img[mask_indices[i][:, 0], mask_indices[i][:, 1], :]*m[i]
        else:
            img[mask_indices[:, 0], mask_indices[:, 1], :] = img[mask_indices[:, 0], mask_indices[:, 1], :] * m
        return img


