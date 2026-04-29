"""
Various dictionaries for working with MobileNet_V3 segmenter labels.
"""

from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = lraspp_mobilenet_v3_large(weights=weights)

# Get the class list (index 0 is background)
classes = weights.meta["categories"]

# Create a mapping dictionary if needed
int_to_class = {
    0: '__background__',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

# 0 -> background
# 1 -> person
# 2 -> animal
# 3 -> car
# 4 -> bus
# 5 -> bike
LABEL_TO_TYPE = {
    0: 0,
    1: 0,
    2: 5,
    3: 0,
    4: 0,
    5: 0,
    6: 4,
    7: 3,
    8: 2,
    9: 0,
    10: 2,
    11: 0,
    12: 2,
    13: 2,
    14: 5,
    15: 1,
    16: 0,
    17: 2,
    18: 0,
    19: 0,
    20: 0
}

# unknown maps to (0, 255, 0) green when used in segmentation_utils
TYPE_TO_COLOR = {
    0: (0, 255, 255),   # background/static (cyan)
    1: (255, 0, 255),     # person (magenta)
    2: (0, 0, 255),       # animal (bright blue)
    3: (255, 0, 0),       # car (bright red)
    4: (255, 165, 0),     # bus (orange)
    5: (255, 255, 0),     # motorcycle/bicycle (yellow)
}

TYPE_TO_DESCRIPTION = {
    0: "background",
    1: "person",
    2: "animal",
    3: "car",
    4: "bus",
    6: "bike"
}

CLASS_TO_COLOR = {
    "car":   [1, 0, 0],      # red
    "bus":   [1, 0.5, 0],    # orange
    "person":[0, 1, 0],      # green
    "bike":  [0, 0, 1],      # blue
    "animal":[1, 0, 1],      # magenta
}