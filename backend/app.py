import os
import psycopg2
import json
import numpy as np
import cv2
import face_recognition # MUDANÇA: Importar a nova biblioteca
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Deixando o CORS mais aberto para testes, pode ser restringido depois se necessário
CORS(app)

# --- O código do banco de dados permanece o mesmo ---
def get_db_connection():
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
            cur.execute("""
            CREATE TABLE IF NOT EXISTS passengers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                document_id VARCHAR(50) UNIQUE NOT NULL,
                embedding JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)
        conn.commit()
        conn.close()
        print("Tabela 'passengers' verificada/criada com sucesso.")

# MUDANÇA: O NumpyEncoder não é mais estritamente necessário, mas podemos manter por segurança
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def process_image_in_memory(photo_file):
    file_bytes = np.fromfile(photo_file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # MUDANÇA: face_recognition espera cores RGB, enquanto OpenCV lê em BGR. Precisamos converter.
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

@app.route('/api/register', methods=['POST'])
def register_passenger():
    if 'photo' not in request.files or 'name' not in request.form or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    name = request.form['name']
    document_id = request.form['document_id']
    photo = request.files['photo']

    try:
        img = process_image_in_memory(photo)
        
        # MUDANÇA: Lógica de representação com face_recognition
        # Retorna uma lista de encodings (um para cada rosto na imagem)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) == 0:
            return jsonify({"error": "Nenhum rosto encontrado na imagem."}), 400
        
        # Pega o encoding do primeiro rosto encontrado
        embedding = face_encodings[0]
        embedding_json = json.dumps(embedding, cls=NumpyEncoder)

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Falha na conexão com o banco de dados"}), 500
        
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO passengers (name, document_id, embedding) VALUES (%s, %s, %s)",
                (name, document_id, embedding_json)
            )
        conn.commit()
        conn.close()

        return jsonify({"message": f"Passageiro {name} registrado com sucesso!"}), 201

    except Exception as e:
        print(f"Erro inesperado: {e}")
        return jsonify({"error": f"Ocorreu um erro interno: {e}"}), 500

@app.route('/api/verify', methods=['POST'])
def verify_passenger():
    if 'photo' not in request.files or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    document_id = request.form['document_id']
    photo = request.files['photo']
    
    try:
        live_img = process_image_in_memory(photo)
        live_encodings = face_recognition.face_encodings(live_img)

        if len(live_encodings) == 0:
            return jsonify({"error": "Nenhum rosto encontrado na imagem para verificação."}), 400
        
        live_embedding = live_encodings[0]

        conn = get_db_connection()
        # ... (código para buscar dados no banco continua o mesmo)
        with conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM passengers WHERE document_id = %s", (document_id,))
            result = cur.fetchone()
            if result:
                passenger_name, db_embedding = result
            else:
                return jsonify({"error": "Passageiro não encontrado"}), 404
        conn.close()

        # MUDANÇA: Lógica de verificação com face_recognition
        # `compare_faces` retorna uma lista de True/False
        matches = face_recognition.compare_faces([db_embedding], live_embedding, tolerance=0.6)
        verified = matches[0]

        # `face_distance` nos dá um valor numérico para a similaridade
        # Distância menor = mais parecido. 0 é idêntico.
        distance = face_recognition.face_distance([db_embedding], live_embedding)[0]
        # Convertendo distância para uma porcentagem de similaridade (forma simples)
        similarity_percentage = (1 - distance) * 100

        return jsonify({
            "verified": bool(verified), # Converte de numpy.bool_ para bool nativo
            "similarity_percentage": f"{similarity_percentage:.2f}%",
            "passenger_name": passenger_name,
            "document_id": document_id
        }), 200

    except Exception as e:
        print(f"Erro inesperado na verificação: {e}")
        return jsonify({"error": f"Ocorreu um erro interno na verificação: {e}"}), 500

if __name__ == '__main__':
    create_table()
    app.run(host='0.0.0.0', port=5001, debug=True)