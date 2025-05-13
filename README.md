# Instructions

The submission consists of the following scripts:

svd.py: creates a co-occurrence matrix and then performs svd on that matrix to obtain word embeddings, which are saved in svd-word-vector.pt

skip_gram.py: creates word embeddings using the skip-gram architecture and saves them in skip-gram-word-vectors.pt (Note: although it was instructed to name this file "skip-gram.py", i could not do so due to **errors caused by importing items from this file to another file using "import"**)

svd-classification.py: trains a RNN model on the SVD word vectors and tests them on the test set

skip-gram-classification.py: trains a RNN model on the skip-gram word vectors and tests them on the test set





svd-classification.py or skip-gram-classification.py can be run directly.

# Important points to note

For SVD, the first 10000 sentenecs were used for training, and for skip-grams, the first 15000 sentences were used for training. More sentences could not be used for SVD as **this would have caused python to kill the script due to excess memory consumption.**

For SVD, words with frequency < 4 were replaced with an unk token, and for skip-grams, words with frequency < 2 were replaced with an unk token. The cutoff frequency had to be lowered for skip-grams as **using 4 as the frequency for skip-grams was found to drastically reduce the number of embeddings and negatively affect the model's performance.**
