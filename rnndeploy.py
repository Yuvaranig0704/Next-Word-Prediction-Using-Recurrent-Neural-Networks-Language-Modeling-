import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="Next Word Predictor", layout="wide")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('next_word_model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please ensure 'next_word_model.h5' exists in the current directory")
        return None

# Load tokenizer with error handling
@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {str(e)}")
        st.error("Please ensure 'tokenizer.pkl' exists in the current directory")
        return None

# Simplified prediction function
def predict_next_word(seed_text, model, tokenizer, seq_len=30, top_n=2):
    try:
        # Tokenize input
        token_list = tokenizer.texts_to_sequences([seed_text.lower()])
        if not token_list or len(token_list[0]) == 0:
            return None, "No recognizable words in input"
        
        # Pad sequence
        token_list = pad_sequences(token_list, maxlen=seq_len, padding='pre')
        
        # Get predictions
        preds = model.predict(token_list, verbose=0)[0]
        top_indices = np.argsort(preds)[-top_n:][::-1]
        
        # Get top 1-2 words
        predictions = []
        for i in top_indices:
            word = tokenizer.index_word.get(i, None)
            if word and word != "<OOV>":
                predictions.append((word, float(preds[i])))
                if len(predictions) >= 2:  # Limit to max 2 predictions
                    break
        
        return predictions, None
    except Exception as e:
        return None, str(e)

# Main app function
def main():
    st.title("🧠 Next Word Prediction")
    st.markdown("""
    Predict the next word using an LSTM neural network trained on Wikipedia text.
    """)
    
    # Load resources
    model = load_model()
    tokenizer = load_tokenizer()
    
    if model is None or tokenizer is None:
        st.stop()
    
    # User input
    seed_text = st.text_input("Start typing your text:", "Machine learning is")
    
    if st.button("Predict Next Word", type="primary"):
        if not seed_text.strip():
            st.warning("Please enter some text first")
        else:
            with st.spinner("Predicting..."):
                predictions, error = predict_next_word(seed_text, model, tokenizer, top_n=2)
                
                if error:
                    st.error(f"Prediction failed: {error}")
                elif not predictions:
                    st.warning("Couldn't make predictions for this input. Try different words.")
                else:
                    st.success("Top predictions:")
                    
                    # Display only 1-2 predictions
                    for word, prob in predictions[:2]:
                        confidence = int(prob * 100)
                        st.write(f"• {word} ({confidence}% confidence)")
                    
                    # Show suggested continuation with top prediction
                    st.markdown(f"""
                    **Suggested continuation:**  
                    {seed_text} {predictions[0][0]}
                    """)

if __name__ == "__main__":
    main()