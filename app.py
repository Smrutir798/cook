import streamlit as st
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# Load the saved model
@st.cache_resource
def load_model():
    with open(r'C:\Users\smrut\env\recipe_recommender_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

model_data = load_model()
tfidf_vectorizer = model_data['tfidf_vectorizer']
tfidf_matrix = model_data['tfidf_matrix']
df = model_data['dataframe']

# Preprocess text
def preprocess_text(text):
    if isinstance(text, str) and text:
        text = text.lower()
        import string
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    return ''

# Calculate similarity
def calculate_similarity(user_ingredients):
    user_ingredients_text = preprocess_text(', '.join(user_ingredients))
    user_tfidf = tfidf_vectorizer.transform([user_ingredients_text])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    return cosine_similarities

# Recommend recipes
def recommend_recipes(user_ingredients, user_prep_time, user_cook_time, top_n=5):
    cosine_similarities = calculate_similarity(user_ingredients)
    df['similarity'] = cosine_similarities

    # Filter recipes based on preparation and cooking time
    filtered_df = df[
        (df['PrepTimeInMins'] <= user_prep_time) &
        (df['CookTimeInMins'] <= user_cook_time)
    ]

    # Sort by similarity and select top N recommendations
    recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations

# Streamlit UI
st.title("Indian Food Recipe Recommender")

st.sidebar.header("Input Preferences")
user_ingredients = st.sidebar.text_input("Enter ingredients (comma-separated):", "onion, tomato, garlic, ginger").split(", ")
user_prep_time = st.sidebar.slider("Preparation Time (minutes):", 0, 120, 30)
user_cook_time = st.sidebar.slider("Cooking Time (minutes):", 0, 120, 45)

if st.sidebar.button("Recommend Recipes"):
    recommendations = recommend_recipes(user_ingredients, user_prep_time, user_cook_time)

    # Display useful data for each recommended recipe
    for index in recommendations.index:
        recipe_info = df.loc[index]
        st.subheader(f"{recipe_info['RecipeName']} ({recipe_info['TranslatedRecipeName']})")
        st.write(f"**Ingredients:** {recipe_info['Ingredients']}")
        st.write(f"**Preparation Time:** {recipe_info['PrepTimeInMins']} mins")
        st.write(f"**Cooking Time:** {recipe_info['CookTimeInMins']} mins")
        st.write(f"**Total Time:** {recipe_info['TotalTimeInMins']} mins")
        st.write(f"**Servings:** {recipe_info['Servings']}")
        st.write(f"**Cuisine:** {recipe_info['Cuisine']}")
        st.write(f"**Course:** {recipe_info['Course']}")
        st.write(f"**Diet:** {recipe_info['Diet']}")
        st.write(f"**Instructions:** {recipe_info['Instructions']}")
        st.write(f"[Recipe URL]({recipe_info['URL']})")
        st.write("---")