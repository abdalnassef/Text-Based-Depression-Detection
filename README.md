#   Text-Based-Depression-Detection
Applying Sentiment analysis on social media data to early detect depression using pretrained model(Bert).
    depression detection.ipynb

```python
Importing libraries
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

# Model
### Loading some sklearn packaces for modelling.
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
```


### for build our model
```python
import tensorflow as tf
import tensorflow_hub as hub
import tokenization
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=False)
```
### Setting some options for general use.
```python
import warnings
plt.style.use('fivethirtyeight')
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')
```
![image](https://user-images.githubusercontent.com/13395314/167274337-b37cfdce-f9b8-4fee-aa46-d70ceffc9fd1.png)

```python
df = pd.read_csv('All_Data.csv')
sns.barplot(x=df['sentiment'].value_counts().index.values, y=df['sentiment'].value_counts())
plt.ylabel('Count')
plt.show()
```
![image](https://user-images.githubusercontent.com/13395314/167274363-44f6c764-de5d-4920-9a4c-83c9e1f72386.png)

```python
tokenizer=Tokenizer()
tokenizer.fit_on_texts(df['URL_HTML_unemoji_reduce_corr'])
word_freq=pd.DataFrame(tokenizer.word_counts.items(),columns=['word','count']).sort_values(by='count',ascending=False)
feature_names=word_freq['word'].values
wc=WordCloud(max_words=400)
wc.generate(' '.join(word for word in feature_names[500:3500] ))
plt.figure(figsize=(15,10))
plt.axis('off')
plt.imshow(wc)
```
![image](https://user-images.githubusercontent.com/13395314/167274408-7d079d98-0fa4-40ab-adb0-752e8001aacf.png)

# Methods
### Emoji

```python
EMOTICONS = {
    u":‑\)":"Happy face",u":\)":"Happy face",    u":-\]":"Happy face",u":\]":"Happy face",    u":-3":"Happy face",    u":3":"Happy face",    u":->":"Happy face",u":>":"Happy face",    u"8-\)":"Happy face",    u":o\)":"Happy face",    u":-\}":"Happy face",u":\}":"Happy face",    u":-\)":"Happy face",    u":c\)":"Happy face",    u":\^\)":"Happy face",u"=\]":"Happy face",    u"=\)":"Happy face",    u":‑D":"Laughing",u":D":"Laughing",    u"8‑D":"Laughing",u"8D":"Laughing",    u"X‑D":"Laughing",u"XD":"Laughing",    u"=D":"Laughing",u"=3":"Laughing",    u"B\^D":"Laughing",u":-\)\)":"Very happy",    u":‑\(":"sad",u":-\(":"sad",    u":\(":"sad",u":‑c":"sad",    u":c":"sad",u":‑<":"sad",    u":<":"sad",u":‑\[":"sad",    u":\[":"sad",u":-\|\|":"sad",    u">:\[":"sad",u":\{":"sad",    u":@":"sad",u">:\(":"sad",    u":'‑\(":"Crying",u":'\(":"Crying",    u":'‑\)":"Tears of happiness",u":'\)":"Tears of happiness",    u"D‑':":"Horror",u"D:<":"Disgust",    u"D:":"Sadness",u"D8":"Great dismay",    u"D;":"Great dismay",    u"D=":"Great dismay",    u"DX":"Great dismay",u":‑O":"Surprise",    u":O":"Surprise",    u":‑o":"Surprise",    u":o":"Surprise",u":-0":"Shock",    u"8‑0":"Yawn",    u">:O":"Yawn",    u":-\*":"Kiss",u":\*":"Kiss",    u":X":"Kiss",    u";‑\)":"Wink",u";\)":"Wink",    u"\*-\)":"Wink",u"\*\)":"Wink",    u";‑\]":"Wink",u";\]":"Wink",    u";\^\)":"Wink",u":‑,":"Wink",    u";D":"Wink",u":‑P":"Tongue sticking out",u":P":"Tongue sticking out",u"X‑P":"Tongue sticking out",u"XP":"Tongue sticking out",u":‑Þ":"Tongue sticking out",u":Þ":"Tongue sticking out",u":b":"Tongue sticking out",u"d:":"Tongue sticking out",u"=p":"Tongue sticking out",u">:P":"Tongue sticking out",u":‑/":"annoyed",u":/":"annoyed",u":-[.]":"annoyed",u">:[(\\\)]":"annoyed",u">:/":"annoyed",u":[(\\\)]":"annoyed",u"=/":"annoyed",u"=[(\\\)]":"annoyed",u":L":"annoyed",u"=L":"annoyed",u":S":"annoyed",u":‑\|":"Straight face",u":\|":"Straight face",u":$":"Embarrassed",u":‑x":"tongue-tied",u":x":"tongue-tied",u":‑#":"tongue-tied",u":#":"tongue-tied",u":‑&":"tongue-tied",u":&":"tongue-tied",u"O:‑\)":"innocent",u"O:\)":"innocent",u"0:‑3":"innocent",u"0:3":"innocent",u"0:‑\)":"innocent",u"0:\)":"innocent",u":‑b":"Tongue sticking out",u"0;\^\)":"innocent",u">:‑\)":"Evil",u">:\)":"Evil",u"\}:‑\)":"Evil",u"\}:\)":"Evil",u"3:‑\)":"Evil",u"3:\)":"Evil",u">;\)":"Evil",u"\|;‑\)":"Cool",u"\|‑O":"Bored",u":‑J":"Tongue-in-cheek",u"#‑\)":"Party all night",u"%‑\)":"confused",u"%\)":"confused",u":-###..":"Being sick",u":###..":"Being sick",u"<:‑\|":"Dump",u"\(>_<\)":"Troubled",u"\(>_<\)>":"Troubled",u"\(';'\)":"Baby",u"\(\^\^>``":"Nervous",u"\(\^_\^;\)":"Nervous",u"\(-_-;\)":"Nervous",u"\(~_~;\) \(・\.・;\)":"Nervous",u"\(-_-\)zzz":"Sleeping",u"\(\^_-\)":"Wink",u"\(\(\+_\+\)\)":"Confused",u"\(\+o\+\)":"Confused",u"\(o\|o\)":"Ultraman",u"\^_\^":"Joyful",u"\(\^_\^\)/":"Joyful",u"\(\^O\^\)／":"Joyful",u"\(\^o\^\)／":"Joyful",u"\(__\)":"respect",u"_\(\._\.\)_":"respect",u"<\(_ _\)>":"respect",u"<m\(__\)m>":"respect",u"m\(__\)m":"respect",u"m\(_ _\)m":"respect",u"\('_'\)":"Sad",u"\(/_;\)":"Sad",u"\(T_T\) \(;_;\)":"Sad",u"\(;_;":"Sad of Crying",u"\(;_:\)":"Sad",u"\(;O;\)":"Sad",u"\(:_;\)":"Sad",u"\(ToT\)":"Sad",u";_;":"Sad",u";-;":"Sad",u";n;":"Sad",u";;":"Sad",u"Q\.Q":"Sad",u"T\.T":"Sad",u"QQ":"Sad",u"Q_Q":"Sad",u"\(-\.-\)":"Shame",u"\(-_-\)":"Shame",u"\(一一\)":"Shame",u"\(；一_一\)":"Shame",u"\(=_=\)":"Tired",u"\(=\^\·\^=\)":"cat",u"\(=\^\·\·\^=\)":"cat",u"=_\^=":"cat",u"\(\.\.\)":"Looking down",u"\(\._\.\)":"Looking down",u"\^m\^":"Giggling with hand covering mouth",u"\(\・\・?":"Confusion",u">\^_\^<":"Normal Laugh",u"<\^!\^>":"Normal Laugh",u"\^/\^":"Normal Laugh",u"\（\*\^_\^\*）" :"Normal Laugh",u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",u"\(^\^\)":"Normal Laugh",u"\(\^\.\^\)":"Normal Laugh",u"\(\^_\^\.\)":"Normal Laugh",u"\(\^_\^\)":"Normal Laugh",u"\(\^\^\)":"Normal Laugh",u"\(\^J\^\)":"Normal Laugh",u"\(\*\^\.\^\*\)":"Normal Laugh",u"\(\^—\^\）":"Normal Laugh",u"\(#\^\.\^#\)":"Normal Laugh",u"\（\^—\^\）":"Waving",u"\(;_;\)/~~~":"Waving",u"\(\^\.\^\)/~~~":"Waving",u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",u"\(T_T\)/~~~":"Waving",u"\(ToT\)/~~~":"Waving",u"\(\*\^0\^\*\)":"Excited",u"\(\*_\*\)":"Amazed",u"\(\*_\*;":"Amazed",u"\(\+_\+\) \(@_@\)":"Amazed",u"\(\*\^\^\)v":"Laughing",u"\(\^_\^\)v":"Laughing",u"\(\(d[-_-]b\)\)":"Listening to music",u'\(-"-\)':"Worried",u"\(ーー;\)":"Worried",u"\(\^0_0\^\)":"Eyeglasses",u"\(\＾ｖ\＾\)":"Happy",u"\(\＾ｕ\＾\)":"Happy",u"\(\^\)o\(\^\)":"Happy",u"\(\^O\^\)":"Happy",u"\(\^o\^\)":"Happy",u"\)\^o\^\(":"Happy",u":O o_O":"Surprised",u"o_0":"Surprised",u"o\.O":"Surpised",u"\(o\.o\)":"Surprised",u"oO":"Surprised",u"\(\*￣m￣\)":"Dissatisfied",u"\(‘A`\)":"Snubbed"
}
```
## Emoji patterns
```python
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
         
```
## Remove Emoji
```python
def remove_emoji(tweet):
    emoji = demoji.findall(tweet)
    
    result = ''
    result1 = ''
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', " ", tweet)    
    for char in tweet:        
        if (len(emoji.get(char, char)) <= 1):
            result += emoji.get(char, char)

    result = emoji_pattern.sub(r'', result)     #remove emojis escapped from tweet    
    

    return result
 ```
    
## Transfer HTML code (<) to < , & to & and .......

```python
def Replace_HTML_codes(text):
    from xml.sax import saxutils as su
    result = su.unescape(text)
    return result
```
## Remove numbers
```python
def remove_num(text):
    text = re.sub(" \d+", " ", text)
    return text
 ```
## Spell Checker
```python
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    text = text.replace("."," ");
    text = unidecode(text)  # Replcae unascii
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
```
## Remove URL (http://ssdfsd ) and @username and change & to and
```python
def Remove_Url_UserName(text): 
    text = re.sub(r"http\S+", "", text , re.IGNORECASE)         #remove url links
    text = re.sub("www.[A-Za-z0-9./]+", ' ', text,re.IGNORECASE)        #remove url links
    text = re.sub('@[^\s]+', ' ', text)     #remove user name
    text = text.replace("&", " and ")       # change & to meaning of sentence
#     text = text.replace("%", " percentage ")       # change & to persentage 
    
    text = re.sub('\n', ' ', text)          #convert to one line only 
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
    return text
```
## Remove (Names,StopWords,punct,Custom stop word)
### Custom stop word
refer to "https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html "
![image](https://user-images.githubusercontent.com/13395314/167274427-85fa7c01-a01f-424e-b6db-6defd264c8f5.png)
```python
custom_stop_word_list=['a','an','are','as','at','by','for','from','if','her','i','me', 'he','him','she', 'himself','they',
                       'you','yours', 'yourselves','themselves','their', 'hereupon','wherein', 'upon',
                       'in','on','onto', 'it','its','of','on','that','the','to','this', 'with','thereafter', 'thence','these',
                       'there', 'sometime','here',  'ourselves', 'when','where','what','whoever',  'whom', 'while','why',
                       'whose', 'whatever','whereas','whenever',  'with', 'who', 'how', 'whither', 'does', 'due',
                       'wherever', 'across', 'somewhere', 'my','mine',  'though', 'itself', 'whence', 'might', 'might', 'we',
                       'as','per', 'whereby', 'since', 'during', 'would', 'such', 'those','which', 'thereby', 'amount', 'at',
                       'into', 'otherwise', 'whether','somehow', 'hence', 'something', 'because', 'meanwhile', 'should', 
                       'still', 'also', 'and','else', 'along', 'another','thru',  'via', 'so', 'after', 
                       'before','may',  'about', 'namely', 'seeming', 'hereby', 'then', 'thereupon','whereafter', 'of', 'to',
                       
                       ## May effect 
                       #'have', 'becoming', 
                        'is','am','be', 'were','was','be','could','being', 'has','are', 
                       'been', 'his',  'us', 'herself',  'do', 'doing', 'both','did', 'had', 
                       ]
contractions = {
"ain't": "are not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how has","i'd": "I would","i'd've": "I would have","i'll": "I will","i'll've": "I will have","i'm": "I am","i've": "I have","isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that had","that'd've": "that would have","that's": "that is","there'd": "there would","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who has","who've": "who have","why's": "why is","why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had / you would","you'd've": "you would have","you'll": "you shall / you will","you'll've": "you shall have / you will have","you're": "you are","you've": "you have"
}
```
```python
def StopWords_punct(text):
    # Remove stop words , punct 
    texts = "Issue in Process "
    text = unidecode(text)  # Replcae unascii
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
                                            #     text_ = Remove_Url_UserName_digits_ConnectLines_ReduceSpaces(text)
    texts = text
    texts = re.sub(' +', ' ', texts)          #convert spaces to one space as names removed 
    
    words = nltk.word_tokenize(texts)
    new_words= [word for word in words if word.isalnum()]


    my_doc_cleaned= [word for word in new_words if word not in custom_stop_word_list]
     
    return my_doc_cleaned
```
## Merge Functions
## Steps without EMOJI
```python
def preprocess_text_removeEmoji(text):
    text = Remove_Url_UserName(text)
    t2 = Replace_HTML_codes(text)
    t3 = remove_emoji(t2)
    t4 = correct_spellings(t3)
    t5 = StopWords_punct(t4)

    text = ' '.join(str(v) for v in t5)
    return text
  ```
## Testing Functions
#### applying preprocessing :-
@neonwonderland somebodye on yt youtube made an mp3 version of each song from the coachella set!!! ;D :) using my cmputr http://www.megaupload.com/?d=EBUP6VVO$#@see go

```python
te = "@neonwonderland somebodye on yt youtube made an mp3 version of each song from the coachella set!!! ;D :) using my cmputr http://www.megaupload.com/?d=EBUP6VVO$#@see go"
Text2 = preprocess_text_removeEmoji(te)
print(Text2)
```
somebody youtube made version each song coachella set using computer go


## Reading Data
===============================================
```python
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.init(num_cpus=4)                # access all cores in CPU
{'node_ip_address': '10.0.0.5',
 'raylet_ip_address': '10.0.0.5',
 'redis_address': '10.0.0.5:6379',
 'object_store_address': '/tmp/ray/session_2022-01-21_23-05-58_926312_18961/sockets/plasma_store',
 'raylet_socket_name': '/tmp/ray/session_2022-01-21_23-05-58_926312_18961/sockets/raylet',
 'webui_url': None,
 'session_dir': '/tmp/ray/session_2022-01-21_23-05-58_926312_18961',
 'metrics_export_port': 50260,
 'node_id': 'd3716e1952be101e3abdd7db5856fe005a2136e9ed7d7fb13b0f3b40'}
data = mpd.read_csv(r"Suicide_Detection.csv" ,header=0, encoding='latin-1' ,  
                  names =["id","content","sentiment"])
data = data[["sentiment" , "content"]]
data.head()
```
![image](https://user-images.githubusercontent.com/13395314/167274481-e5dcadca-38ef-4a69-aa7e-657441edadf1.png)

```python
data.info()
```
![image](https://user-images.githubusercontent.com/13395314/167274529-fa935106-d2f6-44da-9234-b0b3f68c1482.png)
```python
data['sentiment'].value_counts()
```
![image](https://user-images.githubusercontent.com/13395314/167274538-b74af9a1-d80d-4578-9fda-e6a044826f66.png)

```python
data['length']=data['content'].str.len()
data=data[data['length']<2000]
data = data[["sentiment" , "content"]]
data.info()
```
![image](https://user-images.githubusercontent.com/13395314/167274545-f35efb63-093c-435c-9919-14943ebf38c9.png)

```python
data['sentiment'].value_counts()
```
![image](https://user-images.githubusercontent.com/13395314/167274556-9f036353-4663-4280-8e3c-2cadf2e6238b.png)

## Apply Preprocessing
```python
def forAll(df_data):

    df_data['URL_HTML_unemoji']=df_data['content'].apply(Remove_Url_UserName).apply(Replace_HTML_codes).apply(remove_emoji)
    df_data['URL_HTML_unemoji_corr']=df_data['URL_HTML_unemoji'].apply(correct_spellings)
    df_data['URL_HTML_unemoji_corr_Num']=df_data['URL_HTML_unemoji'].apply(remove_num)
    df_data['URL_HTML_unemoji_reduce_corr']=df_data['URL_HTML_unemoji_corr_Num'].apply(StopWords_punct)
    
    # df_data.to_csv('All_Data.csv', encoding='latin-1', index=False)
    return df_data
PreProcess_data = forAll(data)  
PreProcess_data.head()
```
![image](https://user-images.githubusercontent.com/13395314/167274590-5ab922c4-ab4a-47e9-93da-8b5ce527bee6.png)
![image](https://user-images.githubusercontent.com/13395314/167274610-53f391c9-46e4-4a30-8ca0-05c178211338.png)
![image](https://user-images.githubusercontent.com/13395314/167274619-c8b32329-369a-4188-ad40-b1abf0910341.png)
![image](https://user-images.githubusercontent.com/13395314/167274628-d1a57081-1dad-44ed-9cd4-7bfa4003ecde.png)
![image](https://user-images.githubusercontent.com/13395314/167274644-10c746ca-9ede-4290-90bd-0b17599f8cc1.png)

# Model
```python
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

![image](https://user-images.githubusercontent.com/13395314/167274674-f2ae9f0c-7296-486a-917e-5bacee5a3f93.png)

```python
df = pd.read_csv('https://transfer.sh/azv2wW/All_Data.csv')
encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(
        df['URL_HTML_unemoji_reduce_corr'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
    )
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
```
```python
def bert_encode(texts, tokenizer=tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    
def bert_encode_predict(text, tokenizer=tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    
    text = tokenizer.tokenize(text)

    text = text[:max_len-2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_len

    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
X_train = bert_encode(X_train, tokenizer)
def build_model(bert_layer, max_len=128):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
```
```python
model = build_model(bert_layer)
model.summary()
```
![image](https://user-images.githubusercontent.com/13395314/167274694-31e33b1e-6cfa-4613-80ca-4f804b797165.png)

```python
tf.keras.utils.plot_model(model)
```

![image](https://user-images.githubusercontent.com/13395314/167274804-d1853836-d308-4f8e-8012-abd53be5a717.png)

```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/best_weights',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=16, callbacks=[model_checkpoint_callback])
```
![image](https://user-images.githubusercontent.com/13395314/167274766-fd9b27b9-4c3e-4caf-9d71-fd79290a402c.png)

```python
model.save('./my_model')
```
![image](https://user-images.githubusercontent.com/13395314/167274795-1adce7c9-3e16-47ad-8e76-57d21383ad20.png)

# Future Plans
Texts
### some issue
Hashtag may have more importance
image if can get it and describe in it
remove names can do that for english name in space
Reduce Words (diseases , drugs , feelingwords ) getting from dir dictionaries
```python
file ='dictionaries/diseases.txt'
All_diseases = []
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 :
            All_diseases.append(line)
file ='dictionaries/drugs.txt'
All_drugs = {}
with open(file) as f:
    drugs_name = "Stimulant notable stimulants"
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 and line[0]!= '#':
            All_drugs[line] = drugs_name
        
        elif len(line) > 0 :
            text = line.split('/')    
            drugs_name = text[-1]
file ='dictionaries/feelingwords_mapping.txt'
All_feelings = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        text = line.split('\t')
        if len(line) > 0 :
                All_feelings[text[-1]] = text[0]
file ='dictionaries/meds.txt'
All_meds = []
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 :
            All_meds.append(line)
file ='dictionaries/feelingwords_mapping.txt'
All_feelings = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        text = line.split('\t')
        if len(line) > 0 :
                All_feelings[text[-1]] = text[0]
```
## there is spaces in them not Handeled 
```python
def reduce(text):
    T = text.split()
    my_doc_cleaned = [ 'diseases' if word in All_diseases else word for word in T ]
    my_doc_cleaned = [ 'meds' if word in All_meds else word for word in my_doc_cleaned ]
    my_doc_cleaned = [ All_feelings.get(word) if word in All_feelings else word for word in my_doc_cleaned ]
    my_doc_cleaned = [ All_drugs.get(word) if word in All_drugs else word for word in my_doc_cleaned ]
    
    return " ".join(my_doc_cleaned)
# reduce("i am annoyed nialamide can't")
Replace Social shortcut
file ='dictionaries/ShortCutsSocial.txt'
ShortCutsSocial = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n').lower()
        text = line.split('â€“')
        if len(line) > 0 :
                ShortCutsSocial[text[0].strip()] =  text[-1].strip()
def Replace_ShortCut_Social(text):
    T = text.lower().split()
    my_doc_cleaned = [ ShortCutsSocial.get(word) if word in ShortCutsSocial else word for word in T ]    
    return " ".join(my_doc_cleaned)
```
replace Emoji
```python
def replace_emoji(tweet):
    emoji = demoji.findall(tweet)
    
    result = ''
    result1 = ''
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), tweet)    
    tweet = tweet.replace(u"\u2122", '')  # remove ™
    tweet = tweet.replace(u"\u20ac", '')   # remove €
    
    for char in tweet:        
        if (len(emoji.get(char, char)) > 1):
            result +=' ' +emoji.get(char, char).replace(" ", " ")
        else :
            result += emoji.get(char, char)
    result = emoji_pattern.sub(r'', result)     #remove emojis escapped from tweet    
    result = result.replace("-"," ")
    return result
```
Transfer 4 > four and numbers to its string
```python
def replace_numbers_with_string(string):
    import inflect
    items = string.split()
    Trans = inflect.engine()

    for idx, item in enumerate(items):
        try:
            repl = False
            nf = float(item)
            ni = int(nf)  
            repl = Trans.number_to_words(ni)
            items[idx] = str(repl)
        except ValueError:
            if repl != False:
                items[idx] = str(repl)  # when we reach here, item is float
    return " ".join(items)
```

```python
lemmatizer = WordNetLemmatizer() 
def lemma_StopWords_punct(text):
    
    for cont in contractions:
        text = re.sub(u'('+cont+')', " ".join(contractions[cont].replace(",","").split()), text) 
        
    texts = "Issue in Process "
    text = unidecode(text)  # Replcae unascii
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
                                            #     text_ = Remove_Url_UserName_digits_ConnectLines_ReduceSpaces(text)   
    lemmatized_sentence = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text_ = lemmatized_sentence

    texts = text_
    texts = re.sub(' +', ' ', texts)          #convert spaces to one space as names removed 
    words = nltk.word_tokenize(texts)
    new_words= [word for word in words if word.isalnum()]
    

    my_doc_cleaned= [word for word in new_words if word not in custom_stop_word_list]
    
    
    return my_doc_cleaned
```
```python
def preprocess_text(text):
    text = Remove_Url_UserName(text)
    t2 = Replace_HTML_codes(text)
    t3 = replace_emoji(t2)
    t4 = Replace_ShortCut_Social(t3)
    t5 = reduce(t4)    
    t6 = correct_spellings(t5)
    t7 = replace_numbers_with_string(t6)
    t8 = lemma_StopWords_punct(t7)

    text = ' '.join(str(v) for v in t8)
    return text
 ```

```python
te = "@neonwonderland somebodye on yt youtube made an mp3 version of each song from the coachella set!!! ;D :) using my cmputr http://www.megaupload.com/?d=EBUP6VVO$#@see go"
Text2 = preprocess_text(te)
print(Text2)
```
somebody youtube youtube made sound version each song coachella set wink happy face using computer go
 



# Depression_Analysis

[Depression Detection.pdf](https://github.com/abdalnassef/DepressionDetection/files/8639141/Depression.Detection.pdf)

![image](https://user-images.githubusercontent.com/13395314/167110043-e46b7e12-0169-441c-823f-d0e57054752f.png)
