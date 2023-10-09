import glob
import cv2

files = glob.glob("10.1.24.31/replay_417375067/*.png")

for image_path in files:
    print(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    # img = convert_ycbcr_to_rgb(img)
    cv2.imwrite(image_path, img)