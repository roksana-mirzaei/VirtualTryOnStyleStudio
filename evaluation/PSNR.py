import numpy as np
import skimage.io
from skimage.transform import resize


class PSNRCalculator:
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

    @staticmethod
    def calculate_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr, mse

    def compute_psnr(self):
        if self.input_img.shape != self.predicted_img.shape:
            input_img_resized = resize(
                self.input_img, self.predicted_img.shape[:2], preserve_range=True, anti_aliasing=True
            ).astype(self.predicted_img.dtype)
        else:
            input_img_resized = self.input_img

        psnr_value, mse_value = self.calculate_psnr(input_img_resized, self.predicted_img)
        return psnr_value, mse_value

    def __call__(self):
        return self.compute_psnr()
