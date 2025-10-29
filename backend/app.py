import os
import psycopg2
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
CORS(app) # Habilita o CORS para permitir requisições do frontend

# --- Configuração do Banco de Dados Neon Tech ---
def get_db_connection():
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return conn
    except psycopg2.OperationalError as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

# --- Função para criar a tabela se ela não existir ---
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

# --- Helpers ---
class NumpyEncoder(json.JSONEncoder):
    """ Encoder especial para converter arrays NumPy em listas para JSON """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# --- Rotas da API ---

@app.route('/api/register', methods=['POST'])
def register_passenger():
    if 'photo' not in request.files or 'name' not in request.form or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    name = request.form['name']
    document_id = request.form['document_id']
    photo = request.files['photo']

    temp_path = os.path.join("uploads", photo.filename)
    photo.save(temp_path)

    try:
        embedding_objs = DeepFace.represent(img_path=temp_path, model_name='Facenet512', enforce_detection=True)
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
        return jsonify({"error": f"Ocorreu um erro: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/verify', methods=['POST'])
def verify_passenger():
    if 'photo' not in request.files or 'document_id' not in request.form:
        return jsonify({"error": "Dados incompletos"}), 400

    document_id = request.form['document_id']
    photo = request.files['photo']
    
    temp_path = os.path.join("uploads", photo.filename)
    photo.save(temp_path)

    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Falha na conexão com o banco de dados"}), 500
            
        db_embedding = None
        passenger_name = None
        with conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM passengers WHERE document_id = %s", (document_id,))
            result = cur.fetchone()
            if result:
                passenger_name, db_embedding_json = result
                # --- ESTA É A LINHA CORRIGIDA ---
                # O psycopg2 já retorna uma lista Python do JSONB, que é o formato ideal
                db_embedding = db_embedding_json
            else:
                return jsonify({"error": "Passageiro não encontrado"}), 404
        conn.close()

        live_embedding_objs = DeepFace.represent(img_path=temp_path, model_name='Facenet512', enforce_detection=True)
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
        return jsonify({"error": f"Ocorreu um erro: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    create_table()
    app.run(host='0.0.0.0', port=5001, debug=True)