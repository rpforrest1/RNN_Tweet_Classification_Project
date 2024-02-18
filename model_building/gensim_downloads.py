from gensim import downloader

# download pretrained models.
model = downloader.load('glove-wiki-gigaword-100')
model = downloader.load('word2vec-google-news-300')

model = downloader.load('glove-twitter-200')
model = downloader.load('glove-wiki-gigaword-300')
model = downloader.load('fasttext-wiki-news-subwords-300')

print('FINISHED')