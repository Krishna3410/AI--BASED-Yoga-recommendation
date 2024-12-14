# AI-BASED-Yoga-recommendation
# Mood-Based Yoga Session Recommendation

This project aims to provide a personalized yoga session recommendation system based on the user's mood. Using sentiment analysis and machine learning, the system detects the user's emotional state from textual input and suggests appropriate yoga routines to enhance their well-being.

---

## Features
1. **Mood Detection**: Analyzes user input to classify their emotional state into predefined categories.
2. **Yoga Session Recommendations**: Maps detected moods to specific yoga practices tailored to the user’s needs.
3. **Interactive User Interface**: A Streamlit-based app for easy user interaction.

---

## Project Structure
```
AI Yoga Recommendation/
|-- app.py                         # Streamlit app for user interaction
|-- Mood_based_recommendation.py   # Model and recommendation logic
|-- emotion_classifier.pkl         # Trained ML model for mood detection
|-- tfidf_vectorizer.pkl           # TF-IDF vectorizer used for text processing
|-- README.md                      # Project documentation (this file)
```

---

## Approach

### 1. Data Preprocessing
- Cleaned text data by removing punctuation, numbers, and stopwords.
- Normalized text by converting it to lowercase.
- Encoded emotion labels into numeric format.
- Split data into training, validation, and test sets.

### 2. Model Architecture
- **Machine Learning Model**: Logistic Regression trained on TF-IDF features extracted from text data.
- **Text Representation**: Used a TF-IDF vectorizer to transform raw text into numerical form suitable for the model.

### 3. Recommendation Logic
- Defined a mood-to-yoga mapping to connect emotions with specific yoga practices:
  - `happy`: Dynamic sequences like Sun Salutations.
  - `sad`: Gentle poses like Child’s Pose and Forward Folds.
  - `anger`: Relaxation poses like Corpse Pose and Deep Breathing.
  - `stressed`: Meditative sequences with Breathing Exercises.
  - `neutral`: Balanced flows like Warrior Poses.

### 4. Integration
- Built a user-friendly interface with Streamlit to allow users to input their feelings and receive real-time recommendations.

---

## Usage

### Prerequisites
Ensure the following Python libraries are installed:
- `streamlit`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`

## Results
- **Model Accuracy**: Achieved a validation accuracy of over 85% for mood classification.
- **Real-Time Recommendations**: Provides immediate yoga session suggestions based on detected emotions.

---

## Next Steps
1. **Enhancements**:
   - Add support for voice input using speech-to-text.
   - Incorporate more emotion categories and yoga routines.
   - Implement multilingual support.
2. **Deployment**:
   - Host the application on a cloud platform like AWS, Heroku, or Streamlit Cloud for public use.

---

## Acknowledgments
- **Datasets**: Leveraged publicly available datasets for sentiment analysis.
- **Libraries**: Hugging Face Transformers, scikit-learn, Streamlit, and Matplotlib for implementation.



