import re
import numpy as np

def extract_features(review, positive_words: set[str], negative_words: set[str]):
    words = set(re.findall(r'\b\w+\b', review.lower()))
    features = {}
    features['positive_count'] = len(words & positive_words)
    features['negative_count'] = len(words & negative_words)
    features['contains_no'] = int('no' in words)
    features['pronoun_count'] = len(words & {'i', 'me', 'my', 'you', 'your'})
    features['contains_exclamation'] = int('!' in review)
    features['log_length'] = np.log(len(review)+1)
    return features