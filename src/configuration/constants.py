import numpy as np

# CLASS_MAPPING = {
#   "field": {
#     "ball": 1,
#     "field": 2,
#     "cross_l": 3,
#     "cross_t": 3,
#     "cross_x": 3,
#     "line": 3,
#     "circle": 3,
#     "goal": 4,
#   },
#   "robots": {
#     "rightleg": 5,
#     "leftleg": 5,
#     "rightarm": 5,
#     "leftarm": 5,
#     "body": 5,
#     "head": 5,
#   }
# }

# CLASS_TO_LABEL = ["Nothing", "Ball", "Field", "Line", "Goal", "Robot"]

# CLASS_MAPPING = {
#   "field": {
#     "ball": 0,
#     "field": 1,
#     "cross_l": 2,
#     "cross_t": 2,
#     "cross_x": 2,
#     "line": 2,
#     "circle": 2,
#     "goal": 0,
#   },
#   "robots": {
#     "rightleg": 0,
#     "leftleg": 0,
#     "rightarm": 0,
#     "leftarm": 0,
#     "body": 0,
#     "head": 0,
#   }
# }

# CLASS_TO_LABEL = ["Nothing", "Field", "Line"]

# CLASS_MAPPING = {
#   "field": {
#     "ball": 0,
#     "field": 1,
#     "cross_l": 2,
#     "cross_t": 2,
#     "cross_x": 2,
#     "line": 2,
#     "circle": 2,
#     "goal": 0,
#   },
#   "robots": {
#     "rightleg": 3,
#     "leftleg": 3,
#     "rightarm": 3,
#     "leftarm": 3,
#     "body": 3,
#     "head": 3,
#   }
# }

# CLASS_TO_LABEL = ["Nothing", "Field", "Line", "Robot"]

CLASS_MAPPING = {
  "field": {
    "ball": 0,
    "field": 0,
    "cross_l": 1,
    "cross_t": 1,
    "cross_x": 1,
    "line": 1,
    "circle": 1,
    "goal": 0,
  },
  "robots": {
    "rightleg": 2,
    "leftleg": 2,
    "rightarm": 2,
    "leftarm": 2,
    "body": 2,
    "head": 2,
  }
}

CLASS_TO_LABEL = ["Other", "Line", "Robot"]

# 0 is also an class
NUM_CLASSES = len(CLASS_TO_LABEL)

# Downscaling factor applied both to height and width 
# of the input image and output mask
DOWNSCALE_FACTOR = 1

# RGB
NUM_INPUT_CHANNELS = 3

NUM_AUGMENTATION_VARIANTS = 5

CLASS_WEIGHTS = np.ones(NUM_CLASSES, dtype=np.float32)
# CLASS_WEIGHTS = np.array([0.00317216, 0.79575046, 0.00122494, 0.03992792, 0.13479571, 0.0251288 ], dtype=np.float32)
