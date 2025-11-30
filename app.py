import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import resampy
from io import BytesIO
from tensorflow.keras.layers import Layer

# --- Constants ---
SR = 16000
SEGMENT_SECONDS = 10
HOP_SECONDS = 5
EMBEDDING_SIZE = 1024
MAX_LEN = 5 

# --- Custom Layer Definition ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs, mask=None):
        scores = tf.tensordot(inputs, self.W, axes=[[2],[0]])
        if mask is not None:
             scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9 
        weights = tf.nn.softmax(scores, axis=1)
        weights = tf.expand_dims(weights, axis=-1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context
    def get_config(self):
         config = super(AttentionLayer, self).get_config()
         return config

# --- Caching the models ---
@st.cache_resource
def load_yamnet_model():
    return hub.load('yamnet_1')

@st.cache_resource
def load_trained_model():
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        model = tf.keras.models.load_model('yamnet_gtzan_model.h5')
    return model

# --- Helper functions ---
def load_audio_segments(waveform, sr=SR, segment_seconds=SEGMENT_SECONDS, hop_seconds=HOP_SECONDS):
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
    waveform = tf.convert_to_tensor(waveform_segment, dtype=tf.float32)
    scores, embeddings, spec = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)

# --- Main Prediction Function (Full Song Scan) ---
def get_prediction(file_data, yamnet_model, trained_model):
    # 1. Load and resample audio
    waveform, _ = librosa.load(file_data, sr=SR, mono=True)

    # 2. Get segments
    segments = load_audio_segments(waveform)
    if not segments:
        return "Could not process audio (file too short?)"

    # 3. Get embeddings
    seg_embs = []
    for seg in segments:
        emb = compute_segment_embedding(seg, yamnet_model)
        seg_embs.append(emb)

    # 4. Process in batches (Sliding Window)
    all_preds = []
    num_segments = len(seg_embs)
    
    if num_segments <= MAX_LEN:
        pad_count = MAX_LEN - num_segments
        pad = np.zeros((pad_count, EMBEDDING_SIZE), dtype=np.float32)
        if num_segments > 0:
            seq = np.concatenate([np.stack(seg_embs, axis=0), pad], axis=0)
        else:
            seq = pad
        
        seq = np.expand_dims(seq, axis=0).astype(np.float32)
        preds = trained_model.predict(seq)
        all_preds.append(preds[0])
        
    else:
        # Non-overlapping chunks of 5
        stride = MAX_LEN 
        for i in range(0, num_segments, stride):
            chunk = seg_embs[i : i + MAX_LEN]
            if len(chunk) < MAX_LEN:
                pad_count = MAX_LEN - len(chunk)
                pad = np.zeros((pad_count, EMBEDDING_SIZE), dtype=np.float32)
                seq = np.concatenate([np.stack(chunk, axis=0), pad], axis=0)
            else:
                seq = np.stack(chunk, axis=0)
            
            seq = np.expand_dims(seq, axis=0).astype(np.float32)
            preds = trained_model.predict(seq)
            all_preds.append(preds[0])

    # 5. Average the predictions
    if not all_preds:
         return "Error: No predictions made."
         
    avg_preds = np.mean(np.array(all_preds), axis=0)

    labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    probabilities = dict(zip(labels, avg_preds))
    return probabilities


# --- Build the Streamlit App ---
st.set_page_config(layout="wide", page_title="Music Genre Classifier", page_icon="🎵")

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
        font-family: 'Orbitron', sans-serif;
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
                
                probabilities = get_prediction(file_data, yamnet, model)

                if isinstance(probabilities, str):
                    st.error(probabilities)
                else:
                    top_genre = max(probabilities, key=probabilities.get)
                    top_confidence = probabilities[top_genre]
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 2px solid #00d4ff; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);">
                        <h2 style="margin:0;">Detected Genre</h2>
                        <h1 style="font-size: 3em; margin: 10px 0;">{top_genre.upper()}</h1>
                        <h3 style="color: #a0a0a0;">Confidence: {top_confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        'Genre': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    
                    st.subheader("Probability Distribution")
                    st.bar_chart(chart_data.set_index('Genre'), color="#00d4ff")