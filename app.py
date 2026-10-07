import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# ---------------------------------------------------------
# 1. Collect FAQs (Expanded Question & Answer Database)
# ---------------------------------------------------------
faq_data = [
    {
        "question": "What are your operating hours?",
        "answer": "We are open Monday through Friday from 9:00 AM to 6:00 PM."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email."
    },
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day full refund policy for all unused products."
    },
    {
        "question": "Do you accept international payments?",
        "answer": "Yes, we accept major credit cards, debit cards, and PayPal worldwide."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page and follow the email steps."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Yes, free standard shipping is available on all orders over $50."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "Reach out to our team anytime via email at support@example.com."
    },
    {
        "question": "Are there any discount codes for new users?",
        "answer": "Yes! Use code WELCOME10 at checkout to get 10% off your first order."
    },
    {
        "question": "Can I cancel or change my order?",
        "answer": "Orders can be modified or canceled within 1 hour of placement."
    },
    {
        "question": "Do you have physical store locations?",
        "answer": "Currently, we operate purely online to keep prices low for you!"
    }
]

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# ---------------------------------------------------------
# 2. Preprocess Text
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation and word not in stop_words
    ]
    return " ".join(cleaned)

preprocessed_questions = [preprocess(q) for q in questions]

# ---------------------------------------------------------
# 3. Match User Question to FAQs
# ---------------------------------------------------------
def get_answer(user_question):
    cleaned_input = preprocess(user_question)
    if not cleaned_input.strip():
        return "Please type a valid question!"

    all_texts = preprocessed_questions + [cleaned_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_match_index = similarity.argmax()

    if similarity[best_match_index] > 0.2:
        return answers[best_match_index]
    else:
        return "I'm sorry, I don't know the answer to that question yet."

# ---------------------------------------------------------
# 4. Streamlit Chat UI
# ---------------------------------------------------------
st.title("🤖 FAQ Chatbot")
st.write("Ask me anything about store hours, shipping, discounts, or support!")

user_input = st.text_input("Type your question here:")

if user_input:
    response = get_answer(user_input)
    st.success(f"**Bot Response:** {response}")