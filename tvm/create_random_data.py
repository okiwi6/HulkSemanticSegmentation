import numpy as np

image = np.random.randn(1, 3, 480//4, 640//4)
np.savez("random_input", data=image)