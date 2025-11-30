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

# --- Custom CSS for Futuristic Look ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        background-image: linear-gradient(315deg, #0e1117 0%, #1a1c24 74%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif; /* You might need to import this font or use a system alternative */
        color: #00d4ff;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(106, 17, 203, 0.5);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(37, 117, 252, 0.8);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6a11cb, #2575fc);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 1px dashed #00d4ff;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎵 AI Music Genre Classifier")
st.markdown("### *Experience the Future of Sound Analysis*")

# Load models
with st.spinner('Initializing Neural Networks...'):
    yamnet = load_yamnet_model()
    model = load_trained_model()

st.success('System Online. Models Loaded.')

uploaded_file = st.file_uploader("Upload Audio Stream (WAV, MP3)", type=["wav", "mp3", "au"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Audio Stream Detected")
        st.audio(uploaded_file)
    
    with col2:
        if st.button("Analyze Frequency Spectrum"):
            with st.spinner("Processing Neural Pathways..."):
                file_data = BytesIO(uploaded_file.getvalue())
                
                # Get all probabilities
                probabilities = get_prediction(file_data, yamnet, model)

                if isinstance(probabilities, str):
                    st.error(probabilities)
                else:
                    # Find top prediction
                    top_genre = max(probabilities, key=probabilities.get)
                    top_confidence = probabilities[top_genre]
                    
                    # Display Top Result
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 2px solid #00d4ff; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);">
                        <h2 style="margin:0;">Detected Genre</h2>
                        <h1 style="font-size: 3em; margin: 10px 0;">{top_genre.upper()}</h1>
                        <h3 style="color: #a0a0a0;">Confidence: {top_confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Prepare data for chart
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        'Genre': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    
                    # Display Bar Chart
                    st.subheader("Probability Distribution")
                    st.bar_chart(chart_data.set_index('Genre'), color="#00d4ff")