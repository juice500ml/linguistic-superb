import pandas as pd
from datasets import load_dataset
from ipapy import UNICODE_TO_IPA
import panphon.sonority
from panphon import FeatureTable
from panphon.distance import Distance


ds = load_dataset(
        "kalbin/VoxAngeles_phones",
        cache_dir="datasets_cache",
        revision="refs/convert/parquet",
    )
ds = ds["test"]
df = pd.DataFrame(ds)



def place_of_articulation(consonant):
    if consonant in UNICODE_TO_IPA:
        return UNICODE_TO_IPA[consonant].place

    # find the closest phone in UNICODE_TO_IPA then get its place
    min_fed = 1.0
    closest_phone = ''
    dist = Distance()
    for phone in UNICODE_TO_IPA.keys():
        fed = dist.feature_edit_distance(consonant, phone)
        if fed < min_fed:
            min_fed = fed
            closest_phone = phone
    
    return UNICODE_TO_IPA[closest_phone].place

VOWEL_SONORITY = 7
son = panphon.sonority.Sonority()
ft = FeatureTable()
def is_vowel(phone):
    # ft.ipa_segs to convert to normalized decomposed form
    return son.sonority(ft.ipa_segs(phone)[-1]) >= VOWEL_SONORITY


oov = set()
for _, row in df.iterrows():
    if row['phone'] not in UNICODE_TO_IPA:
        oov.add(row['phone'])
        # if row['phone'] in { "k̚\t", 't͡ʃ\u2009' }:
        #     print(row['phone'], row['file'])

for phone in oov:
    if not ft.ipa_segs(phone) or len(ft.ipa_segs(phone)) == 0:
        continue

    try:
        if not is_vowel(phone):
            print(place_of_articulation(phone))
    except:
        print(phone)
