import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (replace with your actual data loading)
df = pd.read_csv('IndianFoodDatasetCSV.csv')  # Replace with your dataset path

# Preprocessing functions (you might need to adjust these)
def preprocess_text(text):
    if isinstance(text, str) and text:
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    else:
        return ''

def extract_ingredients(text):
    if pd.isnull(text):
        return ''
    # Assuming you're using spaCy for ingredient extraction
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    ingredients = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] or token.dep_ == 'compound']
    return ', '.join(ingredients)

# Apply preprocessing to your dataset
df['Ingredients'] = df['TranslatedIngredients'].apply(extract_ingredients)
df['Ingredients'] = df['Ingredients'].apply(preprocess_text)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Ingredients'])

# Recommendation function (adjust as needed)
def calculate_similarity(user_ingredients, user_prep_time, user_cook_time):
    user_ingredients_text = preprocess_text(', '.join(user_ingredients))
    user_tfidf = tfidf_vectorizer.transform([user_ingredients_text])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    
    prep_time_similarity = 1 - abs(df['PrepTimeInMins'] - user_prep_time) / df['PrepTimeInMins'].max()
    cook_time_similarity = 1 - abs(df['CookTimeInMins'] - user_cook_time) / df['CookTimeInMins'].max()
    
    min_length = min(len(cosine_similarities), len(prep_time_similarity), len(cook_time_similarity))
    cosine_similarities = cosine_similarities[:min_length]
    prep_time_similarity = prep_time_similarity[:min_length]
    cook_time_similarity = cook_time_similarity[:min_length]
    
    combined_similarity = (cosine_similarities + prep_time_similarity + cook_time_similarity) / 3
    return combined_similarity

def recommend_recipes(user_ingredients, user_prep_time, user_cook_time, top_n=5):
    combined_similarity = calculate_similarity(user_ingredients, user_prep_time, user_cook_time)
    sorted_indices = combined_similarity.argsort()[::-1]
    top_recommendations = df.iloc[sorted_indices[:top_n], df.columns.get_indexer(['RecipeName', 'TranslatedRecipeName'])].copy()
    return top_recommendations

# Streamlit app
st.title("Recipe Recommender")

user_ingredients = st.text_input("Enter ingredients (comma-separated):")
user_prep_time = st.number_input("Enter prep time (minutes):", min_value=0)
user_cook_time = st.number_input("Enter cook time (minutes):", min_value=0)

if st.button("Get Recommendations"):
    ingredients_list = [ing.strip() for ing in user_ingredients.split(",")]
    recommendations = recommend_recipes(ingredients_list, user_prep_time, user_cook_time)
    st.table(recommendations)