
import string
import re
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from datetime import timedelta


# Load Tokenizer
class DocumentClassificationDataset():
    # Static constant variable
    LABEL2INDEX = {'pelecehan seksual': 0, 'kontak fisik langsung': 1, 'perilaku non-verbal langsung': 2, 'cyber bullying': 3, 'kontak verbal langsung':4, 'perilaku non-verbal tidak langsung':5,}
    INDEX2LABEL = {0: 'pelecehan seksual', 1: 'kontak fisik langsung', 2: 'perilaku non-verbal langsung', 3: 'cyber bullying',4:'kontak verbal langsung',5:'perilaku non-verbal tidak langsung'}
    NUM_LABELS = 6
# Initialize Model Configuration

model_save_path = "indobert_classification_model"
tokenizer_save_path = "indobert_tokenizer"

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

# Set Model ke Mode Evaluasi
model.eval()

app=Flask(__name__)
CORS(app)
host_ip ='0.0.0.0'
def predict_text_bullying(model, tokenizer, text, max_seq_len=512, device='cpu'):
    model.eval()
    torch.set_grad_enabled(False)

    # Tokenize input text
    subwords = tokenizer.encode(text, add_special_tokens=True)[:max_seq_len]

    # Convert to tensor
    subword_tensor = torch.tensor(subwords, dtype=torch.long).unsqueeze(0).to(device)

    # Generate mask
    mask = torch.ones_like(subword_tensor).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(subword_tensor, attention_mask=mask)
        logits = outputs.logits

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Convert index to label
    label = DocumentClassificationDataset.INDEX2LABEL[predicted_label]

    return label,predicted_label


def preprocess_text(text):
    # Menghilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text



# Konfigurasi JWT
app.config["JWT_SECRET_KEY"] = "rahasia-siandung"  # Ganti dengan kunci rahasia Anda
jwt = JWTManager(app)

@app.route('/predict_bullying', methods=['POST'])
@jwt_required()  # Memerlukan token untuk mengakses
def predict_bullying():
    try:
        # Dapatkan teks dari permintaan POST
        text = request.form.get('text', '')

        # Periksa apakah teks tidak kosong
        if not text:
            raise BadRequest("Teks tidak boleh kosong.")
        
        # Preprocess teks
        text = preprocess_text(text)

        # Prediksi pembulian
        predicted_bullying,id = predict_text_bullying(model, tokenizer, text, device='cpu')

        # Custom logic for mapping ids
        id_mapping = {0: 7, 2: 4, 3: 6,4:3}
        id = id_mapping.get(id, id)

        # Kirim kembali hasil prediksi sebagai respons JSON
        return jsonify({'predicted_bullying': predicted_bullying, "id":id}),200

    except BadRequest as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host=host_ip, debug=True)