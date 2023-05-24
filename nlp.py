import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import string

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text for analysis
text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence 
that focuses on the interaction between computers and humans through natural language.
NLP techniques are used to analyze, understand, and generate human language in a valuable way.
"""

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Remove punctuation and convert to lowercase
tokens = [token.lower() for token in tokens if token not in string.punctuation]

# Stopwords removal
stopwords = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stopwords]

# Lemmatization
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Frequency Distribution
fdist = FreqDist(tokens)
print("Most Common Words:", fdist.most_common(5))

# Parts of Speech (POS) Tagging
pos_tags = nltk.pos_tag(tokens)
print("POS Tags:", pos_tags)
