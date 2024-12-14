import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('emotion_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define mood-to-yoga mapping
mood_to_yoga = {
    'happy': 'Dynamic sequences like Sun Salutations',
    'sad': 'Gentle poses like Child’s Pose and Forward Folds',
    'anger': 'Relaxation poses like Corpse Pose and Deep Breathing',
    'stressed': 'Meditative sequences with Breathing Exercises',
    'neutral': 'Balanced flows like Warrior Poses',
}

# Define a list of emotion labels
emotion_labels = ['happy', 'sad', 'anger', 'stressed', 'neutral']

# Recommendation logic
def recommend_yoga(text):
    text_tfidf = vectorizer.transform([text])
    predicted_emotion_numeric = model.predict(text_tfidf)[0]
    emotion_label = emotion_labels[predicted_emotion_numeric]
    yoga_session = mood_to_yoga.get(emotion_label, 'Explore any yoga session of your choice')
    return emotion_label, yoga_session

# Streamlit app
st.title("Mood-Based Yoga Session Recommendation")

# Input area for user feelings
user_input = st.text_area("How are you feeling today?", placeholder="Enter your thoughts...")

# Recommend yoga session when the button is clicked
if st.button("Recommend Yoga Session"):
    if user_input.strip():
        # Call the recommendation logic
        emotion, session = recommend_yoga(user_input)
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Recommended Yoga Session:** {session}")
    else:
        st.write("Please enter your mood or feelings to get a recommendation!")
