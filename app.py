import streamlit as st
import  nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import  pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


ps=PorterStemmer()
def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenization
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  # removing special characters
    text = y[:]
    y.clear()

    # removing stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms=st.text_input("Enter the message")

# 1.Preprocess
transformed_sms=transform_text(input_sms)
# 2.Vectorize
vector_input=tfidf.transform(transformed_sms)
# 3.Predict
result=model.predict(vector_input)[0]
# 4.Display
if result==1:
    st.header("Spam")
else:
    st.header("Not Spam")