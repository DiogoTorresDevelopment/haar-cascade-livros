from app.utils.file_utils import list_images, save_text_file
from app.config.settings import NEGATIVE_PATH
import os

neg_images = list_images(NEGATIVE_PATH)
lines = [os.path.join(NEGATIVE_PATH, os.path.basename(f)).replace("\\", "/") for f in neg_images]
save_text_file(os.path.join(NEGATIVE_PATH, "bg.txt"), "\n".join(lines))