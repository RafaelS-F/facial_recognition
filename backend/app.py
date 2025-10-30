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
# Deixando o CORS aberto para aceitar requisições de qualquer frontend.
# Para produção, você pode restringir a URL da Vercel aqui.
CORS(app) 

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def process_image_in_memory(photo_file):
    file_bytes = np.fromfile(photo_file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

@app.route('/api/register', methods=['POST'])
def register_passenger():
    # (O código desta função permanece o mesmo)
    if 'photo' not in request.files or 'name' not in request.form or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    name = request.form['name']
    document_id = request.form['document_id']
    photo = request.files['photo']

    try:
        img = process_image_in_memory(photo)
        embedding_objs = DeepFace.represent(img_path=img, model_name='Facenet512', enforce_detection=True)
        embedding = embedding_objs[0]['embedding']
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

    except ValueError as e:
        return jsonify({"error": f"Não foi possível detectar um rosto na imagem: {e}"}), 400
    except Exception as e:
        print(f"Erro no registro: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_passenger():
    # (O código desta função permanece o mesmo)
    if 'photo' not in request.files or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    document_id = request.form['document_id']
    photo = request.files['photo']
    
    try:
        live_img = process_image_in_memory(photo)

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Falha na conexão com o banco de dados"}), 500
            
        db_embedding = None
        passenger_name = None
        with conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM passengers WHERE document_id = %s", (document_id,))
            result = cur.fetchone()
            if result:
                passenger_name, db_embedding = result
            else:
                return jsonify({"error": "Passageiro não encontrado"}), 404
        conn.close()

        live_embedding_objs = DeepFace.represent(img_path=live_img, model_name='Facenet512', enforce_detection=True)
        live_embedding = live_embedding_objs[0]['embedding']
        
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
        print(f"Erro inesperado na verificação: {e}")
        return jsonify({"error": f"Ocorreu um erro interno na verificação: {e}"}), 500

if __name__ == '__main__':
    create_table()
    # Railway fornece a porta através da variável de ambiente PORT.
    # O valor padrão 5001 é usado para rodar localmente.
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', port=port)