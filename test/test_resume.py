import unittest
import uuid
from unittest.mock import Mock

import PIL.Image

from data.data_loader import DataLoaderMultiAspect
from data.every_dream import EveryDreamBatch
from data.image_train_item import ImageTrainItem, ImageCaption


def make_test_image_train_item(pathname: str) -> ImageTrainItem:
    caption = ImageCaption(f"caption for {pathname}", 1.0, [], [], 2048, False)
    item = ImageTrainItem(None, caption=caption, aspects=[1.0], pathname=pathname)
    # force wh
    item.target_wh = [512, 512]
    return item


def make_test_image_train_items(count: int) -> [ImageTrainItem]:
    return [make_test_image_train_item(f"image{i:04}.jpg") for i in range(count)]


class MyTestCase(unittest.TestCase):

    def test_resume_state_dlma(self):
        dlma = DataLoaderMultiAspect(image_train_items=make_test_image_train_items(10), seed=555)
        edb = EveryDreamBatch(dlma, tokenizer=Mock(model_max_length=77))
        edb.shuffle(0, 100)
        edb.shuffle(1, 100)
        image_order = [i.pathname for i in edb.image_train_items]

        state_to_resume = dlma.state_dict()

        dlma_2 = DataLoaderMultiAspect(image_train_items=make_test_image_train_items(10), seed=555)
        edb_2 = EveryDreamBatch(dlma, tokenizer=Mock(model_max_length=77))
        pre_resume_state = dlma_2.state_dict()
        self.assertNotEqual(pre_resume_state, state_to_resume)
        dlma_2.load_state_dict(state_to_resume)
        self.assertEqual(state_to_resume, dlma_2.state_dict())
        image_order_2 = [i.pathname for i in edb_2.image_train_items]
        self.assertSequenceEqual(image_order, image_order_2)


        return dlma



    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
