import skimage.io
import skimage.metrics
from skimage.transform import resize


class SSIMCalculator:
    def __init__(self, input_img_path, predicted_img_path):
        """
        Parameters:
        - input_img_path: Path to the input image (str).
        - predicted_img_path: Path to the predicted image (str).
        """
        self.input_img_path = input_img_path
        self.predicted_img_path = predicted_img_path
        self.input_img = skimage.io.imread(input_img_path)
        self.predicted_img = skimage.io.imread(predicted_img_path)

    def compute_similarity(self):
        if self.input_img.shape[:2] != self.predicted_img.shape[:2]:
            input_img_resized = resize(
                self.input_img, self.predicted_img.shape[:2], preserve_range=True, anti_aliasing=True
            ).astype(self.predicted_img.dtype)
        else:
            input_img_resized = self.input_img

        ssim_score = skimage.metrics.structural_similarity(
            input_img_resized, self.predicted_img, channel_axis=-1, win_size=3
        )
        return ssim_score

    def __call__(self):
        return self.compute_similarity()
