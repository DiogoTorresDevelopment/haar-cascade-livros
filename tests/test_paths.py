from app.config.settings import POSITIVE_PATH
import os

print("POSITIVE_PATH:", POSITIVE_PATH)
print("Existe?", os.path.exists(POSITIVE_PATH))