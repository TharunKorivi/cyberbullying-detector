import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import string
from utils.__model_core import predict_bilstm
from download_models import download_models
import nltk
nltk.data.path.append('/opt/render/nltk_data')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
download_models()


LABELS = ['Religion', 'Age', 'Ethnicity', 'Gender', 'Not Bullying']
LABEL_COLORS = {
    'Religion':     '#e67e22',
    'Age':          '#9b59b6',
    'Ethnicity':    '#e74c3c',
    'Gender':       '#3498db',
    'Not Bullying': '#2ecc71',
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t",   " not",    text)
    text = re.sub(r"\'re",   " are",    text)
    text = re.sub(r"\'s",    " is",     text)
    text = re.sub(r"\'d",    " would",  text)
    text = re.sub(r"\'ll",   " will",   text)
    text = re.sub(r"\'t",    " not",    text)
    text = re.sub(r"\'ve",   " have",   text)
    text = re.sub(r"\'m",    " am",     text)
    return text

def strip_all_entities(text):
    text  = text.replace('\r', ' ').replace('\n', ' ').lower()
    text  = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text  = re.sub(r'[^\x00-\x7f]', '', text)
    table = str.maketrans('', '', string.punctuation)
    text  = text.translate(table)
    text  = [w for w in text.split() if w not in stop_words]
    text  = ' '.join(text)
    text  = ' '.join(w for w in text.split() if len(w) < 14)
    return text

def clean_hashtags(tweet):
    t = " ".join(w.strip() for w in
                 re.split(r'#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    return " ".join(w.strip() for w in re.split(r'#|_', t))

def filter_chars(a):
    return ' '.join('' if ('$' in w or '&' in w) else w for w in a.split())

def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

def stemmer(text):
    ps = PorterStemmer()
    return ' '.join(ps.stem(w) for w in nltk.word_tokenize(text))

def deep_clean(text):
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    return text

# ── BiLSTM model definition ───────────────────────────────────────────────────
class BiLSTM_Sentiment_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 lstm_layers, bidirectional, batch_size, dropout):
        super().__init__()
        self.lstm_layers    = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim     = hidden_dim
        self.batch_size     = batch_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * self.num_directions, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        embedded    = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embedded, hidden)
        out         = self.dropout(out[:, -1, :])
        return self.softmax(self.fc(out)), hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.lstm_layers * self.num_directions,
                         batch_size, self.hidden_dim)
        c0 = torch.zeros(self.lstm_layers * self.num_directions,
                         batch_size, self.hidden_dim)
        return (h0, c0)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_bilstm():
    with open('models/config.pkl', 'rb') as f:
        cfg = pickle.load(f)
    with open('models/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_to_int     = {w: i + 1 for i, (w, c) in enumerate(vocab)}
    embedding_matrix = np.load('models/embedding_matrix.npy')
    model = BiLSTM_Sentiment_Classifier(
        cfg['vocab_size'], cfg['embedding_dim'], cfg['hidden_dim'],
        cfg['num_classes'], cfg['lstm_layers'], cfg['bidirectional'],
        cfg['batch_size'], cfg['dropout']
    )
    model.load_state_dict(torch.load('models/state_dict.pt', map_location='cpu'))
    model.eval()
    return model, vocab_to_int, cfg['max_len']

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Cyberbullying Detection")
st.caption("Major Project ")
st.divider()

tweet = st.text_area(
    "Enter tweet",
    placeholder="Type or paste a tweet here...",
    height=120
)

if st.button("Analyse", type="primary"):
    if not tweet.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing... (may take a moment if API is rate limited)"):
            pred_label, probs = predict_bilstm(tweet)

        pred_confidence = float(probs[LABELS.index(pred_label)])
        color           = LABEL_COLORS[pred_label]
        is_bullying     = pred_label != 'Not Bullying'

        st.subheader("Prediction")
        st.markdown(
            f"<div style='padding:18px 24px;border-radius:10px;"
            f"background:{color}22;border-left:6px solid {color};"
            f"font-size:1.3rem;font-weight:700;color:{color};margin-bottom:8px'>"
            f"{'⚠️ Cyberbullying Detected — ' if is_bullying else '✅ '}{pred_label}"
            f"</div>"
            f"<div style='font-size:0.95rem;color:#888;margin-bottom:16px'>"
            f"Confidence: <b>{pred_confidence * 100:.1f}%</b></div>",
            unsafe_allow_html=True
        )

        st.subheader("Confidence scores")
        sorted_pairs = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_pairs:
            st.progress(
                float(prob),
                text=f"{'🔴' if label != 'Not Bullying' else '🟢'} {label}: {prob * 100:.1f}%"
            )