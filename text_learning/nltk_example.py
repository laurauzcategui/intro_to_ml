#!/usr/bin/python

from nltk.corpus import stopwords
sw = stopwords.words("english")

print(sw[0])

print(len(sw))

# using a Stemmer 
from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer("english")

print(st.stem("responsiveness"))
