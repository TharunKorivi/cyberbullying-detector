import os
from huggingface_hub import hf_hub_download

# ── Replace with your Hugging Face username ───────────────────────────────────
HF_USERNAME = "TharunKorivi"
REPO_ID     = f"{HF_USERNAME}/cyberbullying-bilstm"

BILSTM_FILES = [
    "state_dict.pt",
    "vocabulary.pkl",
    "embedding_matrix.npy",
    "config.pkl",
]

BERT_FILES = [
    "pytorch_model.bin",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
]

def download_models():
    # Download BiLSTM model files → models/
    os.makedirs("models", exist_ok=True)
    for filename in BILSTM_FILES:
        dest = os.path.join("models", filename)
        if os.path.exists(dest):
            print(f"  ✅ models/{filename} already present")
            continue
        print(f"  Downloading models/{filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"models/{filename}",
            repo_type="model",
            local_dir=".",
        )
        print(f"  ✅ models/{filename} done")

    # Download BERT files → bert_multiclass/
    os.makedirs("bert_multiclass", exist_ok=True)
    for filename in BERT_FILES:
        dest = os.path.join("bert_multiclass", filename)
        if os.path.exists(dest):
            print(f"  ✅ bert_multiclass/{filename} already present")
            continue
        print(f"  Downloading bert_multiclass/{filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"bert_multiclass/{filename}",
            repo_type="model",
            local_dir=".",
        )
        print(f"  ✅ bert_multiclass/{filename} done")

if __name__ == "__main__":
    download_models()