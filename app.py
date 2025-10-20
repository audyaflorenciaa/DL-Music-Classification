import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import resampy
from io import BytesIO
# Import Layer if you haven't already explicitly (it's in the original imports)
from tensorflow.keras.layers import Layer

# --- Constants from your notebook ---
SR = 16000
SEGMENT_SECONDS = 10
HOP_SECONDS = 5
EMBEDDING_SIZE = 1024
MAX_LEN = 5 # This was the final shape (5, 1024) in your notebook

# --- PASTE THE CUSTOM LAYER DEFINITION HERE ---
# Attention layer for sequence aggregation (simple)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs, mask=None):
        # inputs: (batch, time, features)
        scores = tf.tensordot(inputs, self.W, axes=[[2],[0]])  # (batch, time)
        if mask is not None:
             # Ensure mask operations are compatible
             scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9 
        weights = tf.nn.softmax(scores, axis=1)  # (batch, time)
        weights = tf.expand_dims(weights, axis=-1)  # (batch, time, 1)
        context = tf.reduce_sum(inputs * weights, axis=1)  # (batch, features)
        return context
    # Add get_config for saving/loading compatibility
    def get_config(self):
         config = super(AttentionLayer, self).get_config()
         return config

# --- Caching the models (VERY IMPORTANT) ---
@st.cache_resource
def load_yamnet_model():
    """Loads the YAMNet model from a local file."""
    return hub.load('yamnet_1') # Assuming yamnet_1 is the extracted folder name

@st.cache_resource
def load_trained_model():
    """Loads your custom-trained classifier with custom layer."""
    # --- MODIFICATION HERE ---
    # Tell Keras about the AttentionLayer when loading
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        model = tf.keras.models.load_model('yamnet_gtzan_model.h5')
    return model

# --- Helper functions from your notebook ---

def load_audio_segments(waveform, sr=SR, segment_seconds=SEGMENT_SECONDS, hop_seconds=HOP_SECONDS):
    """
    Load audio, resample to sr, return list of segments.
    (Slightly modified from notebook to accept waveform array instead of path)
    """
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if waveform.size == 0:
        return []
    segments = []
    if len(waveform) <= seg_len:
        if len(waveform) < seg_len:
            waveform = np.pad(waveform, (0, seg_len - len(waveform)))
        segments.append(waveform[:seg_len])
        return segments
    for start in range(0, max(1, len(waveform) - seg_len + 1), hop_len):
        seg = waveform[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments

def compute_segment_embedding(waveform_segment, yamnet_model):
    """
    Computes YAMNet embedding for a single segment.
    (Modified to accept the loaded model as an argument)
    """
    waveform = tf.convert_to_tensor(waveform_segment, dtype=tf.float32)
    scores, embeddings, spec = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)

# --- Main Prediction Function ---
def get_prediction(file_data, yamnet_model, trained_model):
    """Runs the full pipeline: file -> segments -> embeddings -> prediction."""

    # 1. Load and resample audio
    waveform, _ = librosa.load(file_data, sr=SR, mono=True)

    # 2. Get segments
    segments = load_audio_segments(waveform)
    if not segments:
        return "Could not process audio (file too short?)", 0.0 # Return a default confidence

    # 3. Get embeddings
    seg_embs = []
    for seg in segments:
        emb = compute_segment_embedding(seg, yamnet_model)
        seg_embs.append(emb)

    # 4. Pad/Truncate embeddings sequence
    if len(seg_embs) >= MAX_LEN:
        seq = np.stack(seg_embs[:MAX_LEN], axis=0)
    else:
        pad_count = MAX_LEN - len(seg_embs)
        pad = np.zeros((pad_count, EMBEDDING_SIZE), dtype=np.float32)
        seq = np.concatenate([np.stack(seg_embs, axis=0), pad], axis=0)

    # Add batch dimension
    seq = np.expand_dims(seq, axis=0).astype(np.float32)

    # 5. Make prediction
    # These labels must match the order from your notebook's LabelEncoder
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    preds = trained_model.predict(seq)
    prediction_index = np.argmax(preds[0])
    predicted_genre = labels[prediction_index]
    confidence_score = preds[0][prediction_index]

    # Handle the case where the prediction function might return a string message
    if isinstance(predicted_genre, str):
         return predicted_genre, confidence_score
    else:
         # This case shouldn't happen with current logic, but good practice
         return "Error in prediction", 0.0


# --- Build the Streamlit App ---
st.set_page_config(layout="wide")
st.title("🎵 Music Genre Classifier")

# Load models
with st.spinner('Loading models... this may take a moment.'):
    yamnet = load_yamnet_model()
    model = load_trained_model()

st.success('Models loaded!')

uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3", "au"])

if uploaded_file is not None:
    # To play the audio
    st.audio(uploaded_file) # Simpler audio call

    # To make prediction
    if st.button("Classify Genre"):
        with st.spinner("Analyzing audio..."):
            # Use BytesIO to let librosa read the uploaded file from memory
            file_data = BytesIO(uploaded_file.getvalue())

            # Make sure get_prediction always returns two values
            result = get_prediction(file_data, yamnet, model)

            if isinstance(result, str): # Handle error message from get_prediction
                st.error(result)
            else:
                genre, confidence = result
                st.header(f"Predicted Genre: {genre.capitalize()}")
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence:.2%}")