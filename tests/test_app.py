import importlib
import sys
import unittest
from PIL import Image

sys.path.append("/workspaces/freshness-scanner")
app = importlib.import_module("app")


class AppTests(unittest.TestCase):
    def test_preprocess_image_resizes_and_normalizes(self):
        image = Image.new("RGB", (256, 256), color=(255, 0, 0))
        processed = app.preprocess_image(image)

        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertEqual(processed.dtype, "float32")
        self.assertTrue((processed >= 0.0).all() and (processed <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
