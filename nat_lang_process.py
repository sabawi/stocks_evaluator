import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample text document
# document = """
# Natural language processing (NLP) is a field of artificial intelligence concerned with the interaction between computers and humans through natural language. NLP enables computers to understand, interpret, and respond to human language in a valuable way. It involves various tasks such as text classification, sentiment analysis, and question answering.

# Question answering (QA) is a challenging task in NLP. It involves providing accurate answers to questions based on a given context or knowledge base. In this example, we'll train a simple QA model using NLTK and scikit-learn to answer questions based on the provided document.

# Let's get started!
# """

file_path = 'doc1.txt'  # Replace with the actual path to your text file


try:
    with open(file_path, 'r') as file:
        document = file.read()
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
    exit()

# Tokenize the document into sentences
sentences = sent_tokenize(document)

# Preprocess sentences
preprocessed_sentences = []
for sentence in sentences:
    # Tokenize each sentence into words
    words = word_tokenize(sentence)

    # Convert words to lowercase
    words = [word.lower() for word in words]

    # Remove punctuation and stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word.isalnum() and word not in stopwords]

    # Convert words back to sentence
    preprocessed_sentence = ' '.join(words)

    # Append preprocessed sentence
    preprocessed_sentences.append(preprocessed_sentence)

# Check if there are any valid sentences
if not any(preprocessed_sentences):
    print("No valid content to vectorize.")
    exit()

# Vectorize the preprocessed sentences using TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

# User query
# query = "prepared to answer questions"

while True:
    query = input("Enter Query : ")
    # Preprocess and vectorize the query
    query = ' '.join([word.lower() for word in word_tokenize(query) if word.isalnum() and word not in stopwords])
    query_vector = vectorizer.transform([query])

    # Compute cosine similarities between the query vector and document vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the index of the most similar sentence
    most_similar_index = cosine_similarities.argmax()

    # Retrieve the corresponding sentence as the answer
    answer = sentences[most_similar_index]

    # Print the answer
    # print("Question:", query)
    print("Answer:", answer)
