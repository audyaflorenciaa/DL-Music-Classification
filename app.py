import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import resampy
from io import BytesIO
from tensorflow.keras.layers import Layer

SR = 16000
SEGMENT_SECONDS = 10
HOP_SECONDS = 5
EMBEDDING_SIZE = 1024
MAX_LEN = 5 

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

@st.cache_resource
def load_yamnet_model():
    return hub.load('yamnet_1')

@st.cache_resource
def load_trained_model():
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        model = tf.keras.models.load_model('yamnet_gtzan_model.h5')
    return model

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

def get_prediction(file_data, yamnet_model, trained_model):
    waveform, _ = librosa.load(file_data, sr=SR, mono=True)

    segments = load_audio_segments(waveform)
    if not segments:
        return "Could not process audio (file too short?)"

    seg_embs = []
    for seg in segments:
        emb = compute_segment_embedding(seg, yamnet_model)
        seg_embs.append(emb)

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

    if not all_preds:
         return "Error: No predictions made."
         
    avg_preds = np.mean(np.array(all_preds), axis=0)

    labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    probabilities = dict(zip(labels, avg_preds))
    return probabilities


st.set_page_config(layout="wide", page_title="SonicPulse AI", page_icon="🔊")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    /* Animations */
    @keyframes pulse {
        0% { text-shadow: 0 0 10px #00d4ff; }
        50% { text-shadow: 0 0 20px #00d4ff, 0 0 30px #9d00ff; }
        100% { text-shadow: 0 0 10px #00d4ff; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Global Styles */
    .stApp {
        background-color: #050511;
        background-image: 
            radial-gradient(circle at 50% 0%, #1a1a40 0%, transparent 70%),
            radial-gradient(circle at 80% 50%, #2a0a2e 0%, transparent 50%);
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Typography */
    h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem !important;
        background: linear-gradient(90deg, #00d4ff, #9d00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 3s infinite;
        margin-bottom: 0 !important;
    }
    
    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #e0e0e0;
    }

    .subtitle {
        font-size: 1.5rem;
        color: #a0a0ff;
        margin-bottom: 3rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
    }

    /* Neon Button */
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(106, 17, 203, 0.6);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(37, 117, 252, 0.8);
    }

    /* File Uploader - Glass Style */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: border-color 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #00d4ff;
    }

    /* Center Button */
    div.stButton > button {
        display: block;
        margin: 0 auto;
        width: auto;
        min-width: 200px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-top: 20px;'><h1>SONIC PULSE AI</h1></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;' class='subtitle'>Advanced Audio Intelligence System</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #e0e0e0; margin-bottom: 40px; font-size: 1.1rem; max-width: 800px; margin-left: auto; margin-right: auto;'>
    Welcome to <strong>SonicPulse AI</strong>. Unlock the secrets of your music with our advanced neural network technology. 
    Simply upload your audio file (WAV or MP3), and our AI will analyze the frequency patterns to identify the genre with high precision. 
    Experience the future of sound analysis today.
</div>
""", unsafe_allow_html=True)

with st.spinner('Initializing Neural Core...'):
    yamnet = load_yamnet_model()
    model = load_trained_model()

uploaded_file = st.file_uploader("Initialize Audio Stream", type=["wav", "mp3", "au"])

if uploaded_file is not None:
    st.markdown("### 📡 Audio Stream Detected")
    st.audio(uploaded_file)
    
    file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
    st.markdown(f"""
    <div style='margin-top: 15px; margin-bottom: 30px; font-size: 0.9rem; color: #a0a0ff;'>
        <p><strong>FILE:</strong> {file_details['Filename']}</p>
        <p><strong>SIZE:</strong> {file_details['File size']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyze_button = st.button("INITIATE ANALYSIS")
    
    if analyze_button:
        with st.spinner("Scanning Frequencies..."):
            file_data = BytesIO(uploaded_file.getvalue())
            probabilities = get_prediction(file_data, yamnet, model)

            if isinstance(probabilities, str):
                st.error(probabilities)
            else:
                top_genre = max(probabilities, key=probabilities.get)
                top_confidence = probabilities[top_genre]
                
                st.markdown(f"""
                <div class='glass-card' style='text-align: center; border: 2px solid #00d4ff; box-shadow: 0 0 30px rgba(0, 212, 255, 0.2); animation: float 6s ease-in-out infinite; margin-top: 20px;'>
                    <h3 style='margin:0; color: #00d4ff; letter-spacing: 2px;'>PRIMARY CLASSIFICATION</h3>
                    <h1 style='font-size: 4rem; margin: 15px 0; background: linear-gradient(to right, #fff, #b0b0b0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{top_genre.upper()}</h1>
                    <p style='font-size: 1.3rem; color: #a0a0ff;'>Confidence Level: {top_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🧬 Audio DNA Sequence")
                
                import pandas as pd
                chart_data = pd.DataFrame({
                    'Genre': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                })
                
                st.bar_chart(chart_data.set_index('Genre'), color="#9d00ff")