#   Text-Based-Depression-Detection
Applying Sentiment analysis on social media data to early detect depression using pretrained model(Bert).
    depression detection.ipynb

```python
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import re
import demoji
from unidecode import unidecode
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer

import os
import modin.pandas as mpd
import ray


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
```



# Depression_Analysis

[Depression Detection.pdf](https://github.com/abdalnassef/DepressionDetection/files/8639141/Depression.Detection.pdf)

![image](https://user-images.githubusercontent.com/13395314/167110043-e46b7e12-0169-441c-823f-d0e57054752f.png)
