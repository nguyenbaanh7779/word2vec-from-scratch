from dataset.text8 import download_text8, load_text8
from dataset.vocabulary import Vocabulary

def get_training_data(max_tokens: int | None = 1_000_000, min_count: int = 5, subsample_t: float = 1e-3):
    # download and load the dataset
    path = download_text8()
    tokens = load_text8(path, max_tokens=max_tokens)

    print("Downloaded and loaded text8 dataset, number of tokens:", len(tokens))

    # create the vocabulary
    print("Creating vocabulary...")
    voc = Vocabulary(tokens, min_count=min_count, subsample_t=subsample_t)
    print("Vocabulary size:", len(voc.word2idx))

    # encode the tokens with subsampling
    print("Encoding tokens with subsampling...")
    tokens = voc.encode_subsampled(tokens)
    print("Number of tokens after encoding and subsampling:", len(tokens))

    return tokens, voc