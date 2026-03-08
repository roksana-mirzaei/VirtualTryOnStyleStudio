import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch
from tryon.Self_Correction_Human_Parsing import SelfCorrectionHumanParsing
from tryon.DensePose import DensePose

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3,
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11,
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3,
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11,
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'top': ['Left-leg', 'Right-leg'],
    'bottom': ['Right-arm', 'Left-arm', 'Face'],
    'full': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}
PROTECT_CLOTH_PARTS = {
    'top': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'bottom': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'full': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}
MASK_CLOTH_PARTS = {
    'top': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'bottom': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'full': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat', ]
}
MASK_DENSE_PARTS = {
    'top': ['torso', 'big arms', 'forearms'],
    'bottom': ['thighs', 'legs'],
    'full': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}


def select_part(part: Union[str, list],
                           parse: np.ndarray,
                           mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask


def smoothen(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothen = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        smoothen = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | smoothen
    return smoothen


class Masker:
    def __init__(self, densepose_models_path, human_parsing_path, device='cuda'):  # arguments for class

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.densepose_processor = DensePose(densepose_models_path, device)

        self.schp_processor_atr = SelfCorrectionHumanParsing(
            checkpoint_path=os.path.join(human_parsing_path, 'exp-schp-201908301523-atr.pth'), device=device)

        self.schp_processor_lip = SelfCorrectionHumanParsing(
            checkpoint_path=os.path.join(human_parsing_path, 'exp-schp-201908261155-lip.pth'), device=device)

    def run_segmentation_model(self, image_or_path):
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': self.schp_processor_lip(image_or_path)
        }

    @staticmethod
    def concat_segmentation_results(
            densepose_mask: Image.Image,
            schp_lip_mask: Image.Image,
            schp_atr_mask: Image.Image,
            part: str = 'full'
    ):

        if part not in ['top', 'bottom', 'full', 'inner', 'outer']:
            raise ValueError(f"{part} must be one of ['top', 'bottom', 'full', 'inner', 'outer']")

        w, h = densepose_mask.size

        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)

        kernel_size = max(w, h) // 25
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)

        # Strong Protect Area (Hands, Face, Accessory, Feet)
        hands_protect_area = select_part(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & \
                             (select_part(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'],
                                                       schp_atr_mask, ATR_MAPPING) | \
                              select_part(['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'],
                                                       schp_lip_mask, LIP_MAPPING))
        face_protect_area = select_part('Face', schp_lip_mask, LIP_MAPPING)

        strong_protect_area = hands_protect_area | face_protect_area

        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        body_protect_area = select_part(PROTECT_BODY_PARTS[part], schp_lip_mask,
                                                     LIP_MAPPING) | select_part(PROTECT_BODY_PARTS[part],
                                                                                             schp_atr_mask, ATR_MAPPING)
        hair_protect_area = select_part(['Hair'], schp_lip_mask, LIP_MAPPING) | \
                            select_part(['Hair'], schp_atr_mask, ATR_MAPPING)
        cloth_protect_area = select_part(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | \
                             select_part(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING)
        accessory_protect_area = select_part(
            (accessory_parts := ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']),
            schp_lip_mask, LIP_MAPPING) | \
                                 select_part(accessory_parts, schp_atr_mask, ATR_MAPPING)
        weak_protect_area = body_protect_area | cloth_protect_area | hair_protect_area | strong_protect_area | accessory_protect_area

        # Mask Area
        strong_mask_area = select_part(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | \
                           select_part(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
        background_area = select_part(['Background'], schp_lip_mask,
                                                   LIP_MAPPING) & select_part(['Background'],
                                                                                           schp_atr_mask, ATR_MAPPING)
        mask_dense_area = select_part(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=0.25, fy=0.25,
                                     interpolation=cv2.INTER_NEAREST)
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        mask_dense_area = cv2.resize(mask_dense_area.astype(np.uint8), None, fx=4, fy=4,
                                     interpolation=cv2.INTER_NEAREST)

        mask_area = (np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)) | mask_dense_area

        mask_area = smoothen(mask_area * 255) // 255  # Convex Hull to expand the mask area
        mask_area = mask_area & (~weak_protect_area)
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernel_size, kernel_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        mask_area = (mask_area | strong_mask_area) & (~strong_protect_area)
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)

        return Image.fromarray(mask_area * 255)

    def __call__(self, image: Union[str, Image.Image], mask_type: str = "top"):

        if mask_type not in ['top', 'bottom', 'full', 'inner', 'outer']:
            raise ValueError(f"{mask_type} must be one of ['top', 'bottom', 'full', 'inner', 'outer']")

        segmentation_results = self.run_segmentation_model(image)

        mask = self.concat_segmentation_results(
            segmentation_results['densepose'],
            segmentation_results['schp_lip'],
            segmentation_results['schp_atr'],
            part=mask_type,
        )

        return {
            'mask': mask,
            'densepose': segmentation_results['densepose'],
            'schp_lip': segmentation_results['schp_lip'],
            'schp_atr': segmentation_results['schp_atr']
        }


if __name__ == '__main__':
    pass
