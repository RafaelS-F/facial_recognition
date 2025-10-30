import os
import psycopg2
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app) 

# --- Banco de Dados (agora armazenaremos a imagem) ---
def get_db_connection():
    # ... (código do banco de dados permanece o mesmo)
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return conn
    except psycopg2.OperationalError as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

def create_table():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            # MUDANÇA: O campo 'embedding' agora será 'face_image' do tipo BYTEA para guardar a imagem
            cur.execute("""
            CREATE TABLE IF NOT EXISTS passengers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                document_id VARCHAR(50) UNIQUE NOT NULL,
                face_image BYTEA NOT NULL, 
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)
        conn.commit()
        conn.close()
        print("Tabela 'passengers' verificada/criada com sucesso.")

# --- Helpers de Imagem com OpenCV ---
# Carrega o classificador de detecção de rosto do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_and_detect_face(photo_file):
    file_bytes = np.frombuffer(photo_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None # Retorna None se nenhum rosto for encontrado
    
    # Extrai a região do primeiro rosto encontrado
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    return face_roi, file_bytes # Retorna o rosto e os bytes originais da imagem

@app.route('/api/register', methods=['POST'])
def register_passenger():
    if 'photo' not in request.files or 'name' not in request.form or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    name = request.form['name']
    document_id = request.form['document_id']
    photo = request.files['photo']

    # Detecta o rosto para garantir que a imagem é válida
    face_roi, original_image_bytes = process_and_detect_face(photo)
    if face_roi is None:
        return jsonify({"error": "Nenhum rosto detectado na imagem de cadastro."}), 400

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # MUDANÇA: Salva os bytes da imagem original no banco
            cur.execute(
                "INSERT INTO passengers (name, document_id, face_image) VALUES (%s, %s, %s)",
                (name, document_id, psycopg2.Binary(original_image_bytes))
            )
        conn.commit()
        conn.close()
        return jsonify({"message": f"Passageiro {name} registrado com sucesso!"}), 201
    except Exception as e:
        print(f"Erro no registro: {e}")
        return jsonify({"error": "Ocorreu um erro interno durante o registro."}), 500

@app.route('/api/verify', methods=['POST'])
def verify_passenger():
    if 'photo' not in request.files or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    document_id = request.form['document_id']
    photo = request.files['photo']

    try:
        # Pega a imagem ao vivo e detecta o rosto
        live_face_roi, _ = process_and_detect_face(photo)
        if live_face_roi is None:
            return jsonify({"error": "Nenhum rosto detectado na imagem de verificação."}), 400

        # Busca a imagem de cadastro no banco
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT name, face_image FROM passengers WHERE document_id = %s", (document_id,))
            result = cur.fetchone()
            if not result:
                return jsonify({"error": "Passageiro não encontrado."}), 404
            passenger_name, db_image_bytes = result
        conn.close()

        # Detecta o rosto na imagem do banco de dados
        db_face_roi, _ = process_and_detect_face(db_image_bytes)
        if db_face_roi is None:
             return jsonify({"error": "Não foi possível encontrar um rosto na imagem de cadastro."}), 500

        # Treina o modelo LBPH com a imagem do banco
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # O modelo precisa de um ID para cada pessoa, usaremos 1 como padrão
        recognizer.train([db_face_roi], np.array([1]))

        # Tenta reconhecer o rosto da imagem ao vivo
        label, confidence = recognizer.predict(live_face_roi)

        # Avalia o resultado
        # A 'confiança' do LBPH é na verdade uma 'distância'. Menor é melhor.
        # Um bom limiar (threshold) é em torno de 50-80.
        verified = confidence < 80 
        similarity_percentage = max(0, 100 - confidence) # Conversão simples para porcentagem

        return jsonify({
            "verified": bool(verified),
            "similarity_percentage": f"{similarity_percentage:.2f}% (Distância: {confidence:.2f})",
            "passenger_name": passenger_name,
            "document_id": document_id
        }), 200
    except Exception as e:
        print(f"Erro na verificação: {e}")
        return jsonify({"error": "Ocorreu um erro interno durante a verificação."}), 500

if __name__ == '__main__':
    create_table()
    app.run(host='0.0.0.0', port=5001, debug=True)