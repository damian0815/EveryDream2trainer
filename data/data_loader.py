"""
Copyright [2022] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import bisect
import json
import logging
import re
from collections import defaultdict

import math
import copy

import random
from typing import List, Dict
from collections import Counter

from tqdm.auto import tqdm

from data.dataset import DEFAULT_MAX_CAPTION_LENGTH
from data.image_train_item import ImageTrainItem, DEFAULT_BATCH_ID, ImageCaption
import PIL.Image

from utils.first_fit_decreasing import first_fit_decreasing

PIL.Image.MAX_IMAGE_PIXELS = 715827880*4 # prevent decompression bomb errors for very large images (e.g. 64k x 64k, which is 4x the default limit of 32k x 32k)


class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    image_train_items: list of `ImageTrainItem` objects
    seed: random seed
    batch_size: number of images per batch
    """
    def __init__(self, image_train_items: list[ImageTrainItem], seed=555, batch_size=1, chunk_shuffle_batch_size=None,
                 batch_id_dropout_p=0, keep_same_sample_at_different_resolutions_together=False,
                 caption_variants: list[str] = None, expand_caption_variants: bool = False):
        self.seed = seed
        self.batch_size = batch_size
        self.chunk_shuffle_batch_size = chunk_shuffle_batch_size or batch_size
        self.prepared_train_data = image_train_items
        self.caption_variants = caption_variants
        self.expand_caption_variants = expand_caption_variants
        random.Random(self.seed).shuffle(self.prepared_train_data)
        self.prepared_train_data = sorted(self.prepared_train_data, key=lambda img: img.caption.rating())
        self.expected_epoch_size = math.floor(sum([i.multiplier for i in self.prepared_train_data]))
        self.batch_id_dropout_p = batch_id_dropout_p
        self.keep_same_sample_at_different_resolutions_together = keep_same_sample_at_different_resolutions_together
        if self.expected_epoch_size != len(self.prepared_train_data):
            logging.info(f" * DLMA initialized with {len(image_train_items)} source images. After applying multipliers, each epoch will train on at least {self.expected_epoch_size} images.")
        else:
            logging.info(f" * DLMA initialized with {len(image_train_items)} images.")

        self.rating_overall_sum: float = 0.0
        self.ratings_summed: list[float] = []
        self.__update_rating_sums()

    @staticmethod
    def __pick_multiplied_set_helper(items: list[ImageTrainItem], multipliers: dict[str, float], required_count: int, randomizer: random.Random):
        picked_images = []

        # first, collect all images + duplicates for multiplier >= 1
        for iti in items:
            while multipliers[iti.uid] >= 1:
                picked_images.append(iti)
                multipliers[iti.uid] -= 1
                if len(picked_images) >= required_count:
                    break

        remaining = required_count - len(picked_images)

        randomizer.shuffle(items)
        # resolve fractional parts, ensure each is only added max once
        while remaining > 0:
            picked = 0
            for iti in items:
                if randomizer.random() < multipliers[iti.uid]:
                    picked_images.append(iti)
                    multipliers[iti.uid] = 0
                    remaining -= 1
                    picked += 1
                    if remaining <= 0:
                        break
            if picked == 0:
                print(f'ran out of items - {remaining} more needed but no more items to pick from')
                break
        
        return picked_images

    def recompute_expected_epoch_size(self) -> None:
        """Recompute expected_epoch_size after item multipliers have been mutated."""
        self.expected_epoch_size = math.floor(sum(i.multiplier for i in self.prepared_train_data))

    def get_shuffled_image_buckets(self, dropout_fraction: float = 1.0) -> list[ImageTrainItem]:
        """
        Returns the current list of `ImageTrainItem` in randomized order,
        sorted into buckets with same sized images.
        
        If dropout_fraction < 1.0, only a subset of the images will be returned.
        
        If dropout_fraction >= 1.0, repicks fractional multipliers based on folder/multiply.txt values swept at prescan.
        
        :param dropout_fraction: must be between 0.0 and 1.0.
        :return: Randomized list of `ImageTrainItem` objects
        """

        self.seed += 1
        randomizer = random.Random(self.seed)

        multipliers = {i.uid: i.multiplier for i in self.prepared_train_data}
        #data_copy = copy.deepcopy(self.prepared_train_data)
        randomizer.shuffle(self.prepared_train_data)

        if dropout_fraction < 1.0:
            picked_images = self.__pick_random_subset(dropout_fraction, randomizer)
        else:
            picked_images = self.__pick_multiplied_set_helper(self.prepared_train_data, multipliers, required_count=self.expected_epoch_size, randomizer=randomizer)
            for i in picked_images:
                assert multipliers[i.uid] < i.multiplier

        randomizer.shuffle(picked_images)

        buckets = defaultdict(list)
        batch_size = self.batch_size

        def _make_bucket_key(image, batch_id_override: str=None):
            return (image.batch_id if batch_id_override is None else batch_id_override,
                          image.target_wh[0],
                          image.target_wh[1])

        def _add_image_to_appropriate_bucket(image: ImageTrainItem, batch_id_override: str=None):
            #if all(image.caption)
            bucket_key = _make_bucket_key(image, batch_id_override)
            buckets[bucket_key].append(image)

        for image_caption_pair in picked_images:
            image_caption_pair.runt_size = 0
            batch_id_override = DEFAULT_BATCH_ID if randomizer.random() <= self.batch_id_dropout_p else None
            _add_image_to_appropriate_bucket(image_caption_pair, batch_id_override=batch_id_override)

        # expand caption dicts
        pre_expanded_counts = {k: len(v) for k,v in buckets.items()}
        if self.expand_caption_variants:
            print(" * DataLoaderMultiAspect expanding caption dicts into multiple items with different subsets of captions, based on caption_variants: ", self.caption_variants)
            pbar = tqdm(desc="expanding caption dicts", total=sum(len(v) for v in buckets.values()))
            for key, bucket_contents in list(buckets.items()):
                pre_expanded_counts[key] = len(bucket_contents)
                expanded_bucket_contents = []
                for item in bucket_contents:
                    expanded_items = expand_caption_dict(item, caption_variants=self.caption_variants)
                    expanded_bucket_contents.extend(expanded_items)
                    pbar.update()
                buckets[key] = expanded_bucket_contents
                #print(f" - expanded bucket {key} from {len(bucket_contents)} to {len(expanded_bucket_contents)} items")

        # handled named batch runts by demoting them to the DEFAULT_BATCH_ID
        for key, bucket_contents in [(k, b) for k, b in buckets.items() if k[0] != DEFAULT_BATCH_ID]:
            runt_count = len(bucket_contents) % batch_size
            if runt_count == 0:
                continue
            runts = bucket_contents[-runt_count:]
            del bucket_contents[-runt_count:]
            for r in runts:
                _add_image_to_appropriate_bucket(r, batch_id_override=DEFAULT_BATCH_ID)
            if len(bucket_contents) == 0:
                del buckets[key]

        for key, bucket_contents in buckets.items():
            expanded_details = "" if len(bucket_contents)==pre_expanded_counts[key] else "" f" (expanded from {pre_expanded_counts[key]} items)"
            print(" - bucket ", key, " has ", len(bucket_contents), " items" + expanded_details)

        for key in buckets.keys():
            random.shuffle(buckets[key])

        # handle remaining runts by taking from unpicked items, and/or randomly duplicating picked items
        for key, bucket_contents in buckets.items():
            if len(bucket_contents) % batch_size != 0:
                assert key[0] == DEFAULT_BATCH_ID, "there should be no more runts in named batches"

                required_count = batch_size - len(bucket_contents) % batch_size
                unpicked_images_preshuffled = [
                    i
                    for i in self.prepared_train_data
                    if _make_bucket_key(i) == key and multipliers[i.uid] > 0
                ]
                topup = self.__pick_multiplied_set_helper(unpicked_images_preshuffled, multipliers, required_count=required_count, randomizer=randomizer)
                bucket_contents.extend(topup)

                # still runts?
                final_truncate_count = len(bucket_contents) % batch_size
                if final_truncate_count > 0:
                    # we weren't able to fill all runts from unpicked images, so duplicate existing items
                    logging.warning(f"After top-up from unpicked images, bucket {key} with {len(bucket_contents)} still has {final_truncate_count} runts. These will be filled by duplicating existing items in the bucket, which may cause some overfitting. To avoid this, consider adding more items at aspect ratio {key} or reduce your batch count.")
                    runt_bucket_start_offset = len(bucket_contents) - final_truncate_count
                    non_runts = bucket_contents.copy()
                    for _ in range(batch_size - final_truncate_count):
                        bucket_contents.append(random.choice(non_runts))
                    for i in range(batch_size):
                        item = copy.deepcopy(bucket_contents[runt_bucket_start_offset + i])
                        item.runt_size = final_truncate_count
                        bucket_contents[runt_bucket_start_offset + i] = item
            assert len(bucket_contents) % batch_size == 0
            assert all(i.target_wh == bucket_contents[0].target_wh for i in bucket_contents), "mixed aspect ratios in a bucket - this shouldn't happen"

        items_by_batch_id = collapse_buckets_by_batch_id(buckets)
        # at this point items have a partially deterministic order
        # (in particular: rarer aspect ratios are more likely to cluster at the end due to stochastic sampling)
        # so we shuffle them to mitigate this, using chunked_shuffle to keep batches with the same aspect ratio together
        items_by_batch_id = {k: chunked_shuffle(v, chunk_size=batch_size, randomizer=randomizer)
                             for k,v in items_by_batch_id.items()}
        if not items_by_batch_id:
            raise RuntimeError("No images available after applying dropout and multipliers. Check your dataset and multiplier settings.")
        # paranoia: verify that this hasn't fucked up the aspect ratio batching
        for items in items_by_batch_id.values():
            batches = chunk_list(items, chunk_size=batch_size)
            for batch in batches:
                target_wh = batch[0].target_wh
                assert all(target_wh == i.target_wh for i in batch[1:]), "mixed aspect ratios in a batch - this shouldn't happen"

        # handle batch_id
        # unlabelled data (no batch_id) is in batches labelled DEFAULT_BATCH_ID.
        items = flatten_buckets_preserving_named_batch_adjacency(items_by_batch_id,
                                                                   batch_size=batch_size)

        items = chunked_shuffle(items, chunk_size=self.chunk_shuffle_batch_size, randomizer=randomizer)

        if self.keep_same_sample_at_different_resolutions_together:
            items = reorder_same_sample_different_resolution_adjacency(items, chunk_size=self.chunk_shuffle_batch_size, randomizer=randomizer)

        wh_counts = Counter([tuple(i.target_wh) for i in items])
        print("Final aspect ratio distribution:")
        for wh, count in wh_counts.items():
            print(f" - {wh}: {count} images, {(count/len(items))*100:.2f}% of total")

        return items


    def __pick_random_subset(self, dropout_fraction: float, picker: random.Random) -> list[ImageTrainItem]:
        """
        Picks a random subset of all images
        - The size of the subset is limited by dropout_faction
        - The chance of an image to be picked is influenced by its rating. Double that rating -> double the chance
        :param dropout_fraction: must be between 0.0 and 1.0
        :param picker: seeded random picker
        :return: list of picked ImageTrainItem
        """

        prepared_train_data = self.prepared_train_data.copy()
        ratings_summed = self.ratings_summed.copy()
        rating_overall_sum = self.rating_overall_sum

        num_images = len(prepared_train_data)
        num_images_to_pick = math.ceil(num_images * dropout_fraction)
        num_images_to_pick = max(min(num_images_to_pick, num_images), 0)

        # logging.info(f"Picking {num_images_to_pick} images out of the {num_images} in the dataset for drop_fraction {dropout_fraction}")

        picked_images: list[ImageTrainItem] = []
        while num_images_to_pick > len(picked_images):
            # find random sample in dataset
            point = picker.uniform(0.0, rating_overall_sum)
            pos = min(bisect.bisect_left(ratings_summed, point), len(prepared_train_data) -1 )

            # pick random sample
            picked_image = prepared_train_data[pos]
            picked_images.append(picked_image)

            # kick picked item out of data set to not pick it again
            rating_overall_sum = max(rating_overall_sum - picked_image.caption.rating(), 0.0)
            ratings_summed.pop(pos)
            prepared_train_data.pop(pos)

        unpicked_images = [i for i in self.prepared_train_data if i not in picked_images]
        return picked_images, unpicked_images

    def __update_rating_sums(self):        
        self.rating_overall_sum: float = 0.0
        self.ratings_summed: list[float] = []
        for item in self.prepared_train_data:
            self.rating_overall_sum += item.caption.rating()
            self.ratings_summed.append(self.rating_overall_sum)


def chunk_list(l: List, chunk_size) -> List:
    num_chunks = int(math.ceil(float(len(l)) / chunk_size))
    return [l[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

def unchunk_list(chunked_list: List):
    return [i for c in chunked_list for i in c]

def collapse_buckets_by_batch_id(buckets: dict[tuple, list[ImageTrainItem]]) -> dict[str, list[ImageTrainItem]]:
    batch_ids = [k[0] for k in buckets.keys()]
    items_by_batch_id = {}
    for batch_id in batch_ids:
        items_by_batch_id[batch_id] = unchunk_list([b for bucket_key,b in buckets.items() if bucket_key[0] == batch_id])
    return items_by_batch_id

def flatten_buckets_preserving_named_batch_adjacency(items_by_batch_id: Dict[str, List[ImageTrainItem]],
                                                       batch_size: int) -> List[ImageTrainItem]:
    # precondition: items_by_batch_id has no incomplete batches
    assert(all((len(v) % batch_size)==0 for v in items_by_batch_id.values()))
    # ensure we don't mix up aspect ratios by treating each chunk of batch_size images as
    # a single unit to pass to first_fit_decreasing()
    filler_items = chunk_list(items_by_batch_id.get(DEFAULT_BATCH_ID, []), batch_size)
    custom_batched_items = [chunk_list(v, batch_size) for k, v in items_by_batch_id.items() if k != DEFAULT_BATCH_ID]
    neighbourly_chunked_items = first_fit_decreasing(custom_batched_items,
                                                     batch_size=1,
                                                     filler_items=filler_items)

    items: List[ImageTrainItem] = unchunk_list(neighbourly_chunked_items)
    return items

def chunked_shuffle(l: List, chunk_size: int, randomizer: random.Random) -> List:
    """
    Shuffles l in chunks, preserving the chunk boundaries and the order of items within each chunk.
    If the last chunk is incomplete, it is not shuffled (i.e. preserved as the last chunk)
    """
    if len(l) == 0:
        return []

    # chunk by effective batch size
    chunks = chunk_list(l, chunk_size)
    # preserve last chunk as last if it is incomplete
    last_chunk = None
    if len(chunks[-1]) < chunk_size:
        last_chunk = chunks.pop(-1)
    randomizer.shuffle(chunks)
    if last_chunk is not None:
        chunks.append(last_chunk)
    l = unchunk_list(chunks)
    return l


def reorder_same_sample_different_resolution_adjacency(l: list[ImageTrainItem], chunk_size: int, randomizer: random.Random) -> list[ImageTrainItem]:

    # chunk by effective batch size
    chunks: list[list[ImageTrainItem]] = chunk_list(l, chunk_size)
    unused_chunks = set(range(len(chunks)))

    in_chunk = defaultdict(list)
    for chunk_index, chunk in enumerate(chunks):
        for item in chunk:
            item: ImageTrainItem
            in_chunk[item.pathname].append(chunk_index)
    chunk_paths_sets: list[set[str]] = [
        {item.pathname for item in chunk}
        for chunk in chunks
    ]
    chunk_resolutions: list[int] = [
        chunk[0].target_wh[0] * chunk[0].target_wh[1]
        for chunk in chunks
    ]

    result_chunk_indices = [0]
    unused_chunks.remove(0)

    with tqdm(desc='finding resolution adjacency', total=len(chunks)) as pbar:
        while unused_chunks:
            pbar.update()
            last_chunk_index = result_chunk_indices[-1]

            last_paths = chunk_paths_sets[last_chunk_index]
            last_resolution = chunk_resolutions[last_chunk_index]

            candidates = sorted(
                [u for u in unused_chunks if chunk_resolutions[u] != last_resolution],
                key=lambda chunk_index: len(last_paths.intersection(chunk_paths_sets[chunk_index])),
                reverse=True
            )
            if len(candidates) == 0:
                # fallback
                selection = unused_chunks.pop()
            else:
                selection = candidates[0]
                unused_chunks.remove(selection)
            result_chunk_indices.append(selection)

    # reverse the list: we want the mismatched chunks first
    result_chunks = [chunks[i] for i in reversed(result_chunk_indices)]
    #print("before _shuffle_no_matches:")
    #_print_path_match_sequences(result_chunks)
    matches = _get_path_match_with_next_count(result_chunks)
    counter = Counter(matches)
    print("Number of batches with elements in common with next:", counter)

    # now distribute the mismatched chunks amongst all the rest, so that they don't clump at the start/end
    result_chunks = _shuffle_no_matches(result_chunks, randomizer=randomizer)
    #print("after _shuffle_no_matches:")
    #_print_path_match_sequences(result_chunks)

    matches = _get_path_match_with_next_count(result_chunks)
    counter = Counter(matches)
    print("After shuffling:", counter)

    return unchunk_list(result_chunks)

def _shuffle_no_matches(chunks: list[list[ImageTrainItem]], randomizer: random.Random) -> list[list[ImageTrainItem]]:
    """ shuffle chunks with no matches throughout the whole epoch,
    so that we don't end up with a chunk of no-match at the start/end
    """
    path_match_with_next_count = _get_path_match_with_next_count(chunks)
    no_matches_chunks = [
        chunks[i] for i, count in enumerate(path_match_with_next_count)
        if count == 0
    ] + [chunks[-1]]
    matches_chunks = [
        chunks[i] for i, count in enumerate(path_match_with_next_count)
        if count > 0
    ]

    for no_match_chunk in no_matches_chunks:
        random_index = randomizer.randint(0, len(matches_chunks))
        matches_chunks.insert(random_index, no_match_chunk)

    return matches_chunks


def _get_path_match_with_next_count(chunks: list[list[ImageTrainItem]]) -> list[int]:
    path_match_with_next_count = [
        len(
            set([item.pathname for item in chunks[chunk_index]]).intersection(
                set([item.pathname for item in chunks[chunk_index+1]])
           )
        )
        for chunk_index in range(0, len(chunks)-1)
    ]
    return path_match_with_next_count

def _print_path_match_sequences(chunks):
    path_match_with_next_count = _get_path_match_with_next_count(chunks)

    current_value, current_value_count = None, 0
    for (i, chunk) in enumerate(chunks):
        if i == len(chunks)-1:
            break
        path_match_count = path_match_with_next_count[i]
        if current_value != path_match_count:
            if current_value is not None:
                print("sequence of ", current_value_count, "x", current_value)
            current_value = path_match_count
            current_value_count = 1
        else:
            current_value_count += 1
    print("sequence of ", current_value_count, "x", current_value)


def _find_actual_variant_keys(caption_keys: list[str], patterns: list[str]) -> list[str]:
    regexes = { pattern: re.compile(r'^'+pattern+r'$', re.IGNORECASE) for pattern in patterns }
    def _matches_variant(key: str, pattern: str) -> bool:
        return regexes[pattern].match(key) is not None
    variant_buckets = {pattern: set(k for k in caption_keys if _matches_variant(k, pattern))
                       for pattern in patterns}

    # take all the keys with exactly one match
    matching_keys = set(next(iter(keys)) for pattern, keys in variant_buckets.items() if len(keys)==1)
    for pattern, keys in variant_buckets.items():
        remaining_keys = keys.difference(matching_keys)
        if remaining_keys:
            matching_keys.add(random.choice(list(remaining_keys)))

    return list(matching_keys)


def expand_caption_dict(item: ImageTrainItem, caption_variants: list[str]) -> list[ImageTrainItem]:
    if not caption_variants:
        return [item]
    if not item.caption.get_caption().startswith('<<json>>'):
        return [item]

    caption_data = json.loads(item.caption.get_caption().replace('<<json>>', ''))
    all_keys = list(caption_data.keys())
    base_keys = _find_actual_variant_keys(all_keys, patterns=caption_variants)
    others = list(set(caption_data.keys()).difference(set(base_keys)))
    if len(base_keys) == 0:
        base_keys.extend(random.sample(others, k=min(len(others), len(caption_variants))))

    expanded_items = []
    random.shuffle(base_keys)
    number_of_captions_per_dict = math.ceil(len(all_keys) / len(base_keys))
    available_keys = list(set(all_keys).difference(set(base_keys)))
    for base_key in base_keys:
        new_item = item.copy_with_new_uid()

        if len(base_keys) > 0:
            other_keys = random.sample(available_keys, k=min(len(available_keys), number_of_captions_per_dict-1))
        else:
            other_keys = list(available_keys)
        for k in other_keys:
            available_keys.remove(k)
        this_keys = [base_key] + other_keys

        caption = "<<json>>" + json.dumps({k: caption_data[k] for k in this_keys})
        new_item.caption = ImageCaption(
            main_prompt=caption,
            rating=1.0,
            tags=[],
            tag_weights=[],
            max_target_length=DEFAULT_MAX_CAPTION_LENGTH,
            use_weights=False)
        expanded_items.append(new_item)
    return expanded_items


