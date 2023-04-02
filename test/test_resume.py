import copy
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

    def test_resume_state_dlma_and_edb(self):
        image_train_items = make_test_image_train_items(10)
        dlma = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb = EveryDreamBatch(dlma, tokenizer=Mock(model_max_length=77))
        edb.shuffle(0, 100)
        edb.shuffle(1, 100)

        dlma_state_to_resume = dlma.state_dict()
        edb_state_to_resume = edb.state_dict()

        dlma_2 = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb_2 = EveryDreamBatch(dlma_2, tokenizer=Mock(model_max_length=77))
        pre_resume_state = dlma_2.state_dict()
        self.assertNotEqual(pre_resume_state, dlma_state_to_resume)
        dlma_2.load_state_dict(dlma_state_to_resume)
        self.assertEqual(dlma_state_to_resume, dlma_2.state_dict())

        edb_2.load_state_dict(edb_state_to_resume)
        self.assertEqual(edb_state_to_resume, edb_2.state_dict())
        self.assertEqual(edb.state_dict(), edb_2.state_dict())
        self.assertEqual(dlma.state_dict(), dlma_2.state_dict())
        self.assertSequenceEqual([i.pathname for i in edb.image_train_items],
                                 [i.pathname for i in edb_2.image_train_items])

        edb.shuffle(2, 100)
        edb_2.shuffle(2, 100)
        self.assertSequenceEqual([i.pathname for i in edb.image_train_items],
                                 [i.pathname for i in edb_2.image_train_items])

        return dlma


    def test_resume_state_dlma_and_edb_missing_item(self):
        image_train_items = make_test_image_train_items(10)
        dlma = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb = EveryDreamBatch(dlma, tokenizer=Mock(model_max_length=77))
        dlma_state_to_resume = dlma.state_dict()
        edb_state_to_resume = edb.state_dict()

        del image_train_items[5]
        dlma_2 = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb_2 = EveryDreamBatch(dlma_2, tokenizer=Mock(model_max_length=77))
        dlma_2.load_state_dict(dlma_state_to_resume)
        with self.assertRaises(ValueError):
            edb_2.load_state_dict(edb_state_to_resume)

    def test_resume_state_dlma_and_edb_extra_item(self):
        image_train_items = make_test_image_train_items(10)
        dlma = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb = EveryDreamBatch(dlma, tokenizer=Mock(model_max_length=77))
        dlma_state_to_resume = dlma.state_dict()
        edb_state_to_resume = edb.state_dict()

        image_train_items += make_test_image_train_items(2)
        dlma_2 = DataLoaderMultiAspect(image_train_items=copy.deepcopy(image_train_items), seed=555)
        edb_2 = EveryDreamBatch(dlma_2, tokenizer=Mock(model_max_length=77))
        dlma_2.load_state_dict(dlma_state_to_resume)
        with self.assertRaises(ValueError):
            edb_2.load_state_dict(edb_state_to_resume)



if __name__ == '__main__':
    unittest.main()
