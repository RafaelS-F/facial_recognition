import os
import psycopg2
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
# Usando uma configuração de CORS mais aberta para simplicidade, pode ser restringida se necessário.
CORS(app) 

# --- Banco de Dados (Voltando a usar a coluna 'embedding') ---
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
            # MUDANÇA: Voltando para a estrutura de tabela original
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

# --- Helpers ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def process_image_in_memory(photo_file):
    file_bytes = np.frombuffer(photo_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # DeepFace espera o formato BGR, que é o padrão do OpenCV, então não precisamos converter.
    return img

# --- Rotas da API com DeepFace ---
@app.route('/api/register', methods=['POST'])
def register_passenger():
    if 'photo' not in request.files or 'name' not in request.form or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    name = request.form['name']
    document_id = request.form['document_id']
    photo = request.files['photo']

    try:
        img = process_image_in_memory(photo)
        
        # Gera o embedding com DeepFace
        embedding_objs = DeepFace.represent(img_path=img, model_name='Facenet512', enforce_detection=True)
        embedding = embedding_objs[0]['embedding']
        embedding_json = json.dumps(embedding, cls=NumpyEncoder)

        conn = get_db_connection()
        with conn.cursor() as cur:
            # Salva o embedding no banco de dados
            cur.execute(
                "INSERT INTO passengers (name, document_id, embedding) VALUES (%s, %s, %s)",
                (name, document_id, embedding_json)
            )
        conn.commit()
        conn.close()
        return jsonify({"message": f"Passageiro {name} registrado com sucesso!"}), 201
    except ValueError as e:
        return jsonify({"error": f"Não foi possível detectar um rosto na imagem: {e}"}), 400
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
        live_img = process_image_in_memory(photo)
        
        # Gera o embedding da imagem ao vivo
        live_embedding_objs = DeepFace.represent(img_path=live_img, model_name='Facenet512', enforce_detection=True)
        live_embedding = live_embedding_objs[0]['embedding']

        # Busca o embedding do banco de dados
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM passengers WHERE document_id = %s", (document_id,))
            result = cur.fetchone()
            if not result:
                conn.close()
                return jsonify({"error": "Passageiro não encontrado."}), 404
            passenger_name, db_embedding = result
        conn.close()

        # Compara os dois embeddings com DeepFace
        result = DeepFace.verify(
            img1_path=live_embedding, 
            img2_path=db_embedding, 
            model_name='Facenet512',
            distance_metric='cosine'
        )

        similarity_percentage = (1 - result['distance']) * 100

        return jsonify({
            "verified": result['verified'],
            "similarity_percentage": f"{similarity_percentage:.2f}%",
            "passenger_name": passenger_name,
            "document_id": document_id
        }), 200
    except ValueError as e:
        return jsonify({"error": f"Não foi possível detectar um rosto na imagem: {e}"}), 400
    except Exception as e:
        print(f"Erro na verificação: {e}")
        return jsonify({"error": "Ocorreu um erro interno durante a verificação."}), 500

# --- INICIALIZAÇÃO DA APLICAÇÃO ---
create_table()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)