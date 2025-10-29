import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import Webcam from 'react-webcam';

const Register = () => {
    const [name, setName] = useState('');
    const [documentId, setDocumentId] = useState('');
    const [photo, setPhoto] = useState(null);
    const [preview, setPreview] = useState(null);
    const [useWebcam, setUseWebcam] = useState(false);
    const [message, setMessage] = useState({ type: '', text: '' });
    
    const webcamRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setPhoto(file);
            setPreview(URL.createObjectURL(file));
            setUseWebcam(false);
        }
    };

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setPreview(imageSrc);
        // Converter base64 para Blob
        fetch(imageSrc)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
                setPhoto(file);
            });
        setUseWebcam(false);
    }, [webcamRef]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!name || !documentId || !photo) {
            setMessage({ type: 'error', text: 'Por favor, preencha todos os campos e selecione uma foto.' });
            return;
        }

        const formData = new FormData();
        formData.append('name', name);
        formData.append('document_id', documentId);
        formData.append('photo', photo);

        try {
            // --- LINHA CORRIGIDA ---
            const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/register`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setMessage({ type: 'success', text: response.data.message });
            // Limpar formulário
            setName('');
            setDocumentId('');
            setPhoto(null);
            setPreview(null);
        } catch (error) {
            const errorMsg = error.response ? error.response.data.error : 'Erro ao conectar ao servidor.';
            setMessage({ type: 'error', text: `Falha no registro: ${errorMsg}` });
        }
    };

    return (
        <div>
            <h2>Cadastro de Novo Passageiro</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Nome Completo</label>
                    <input type="text" className="form-input" value={name} onChange={(e) => setName(e.target.value)} />
                </div>
                <div className="form-group">
                    <label>Número do Documento</label>
                    <input type="text" className="form-input" value={documentId} onChange={(e) => setDocumentId(e.target.value)} />
                </div>
                
                <div className="form-group">
                    <label>Foto do Rosto</label>
                    <div className="photo-options">
                        <button type="button" className="nav-button" onClick={() => {setUseWebcam(true); setPreview(null);}}>Usar Câmera</button>
                        <span>OU</span>
                        <input type="file" accept="image/*" onChange={handleFileChange} />
                    </div>
                </div>

                {useWebcam && (
                    <div className="webcam-container">
                        <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" />
                        <button type="button" className="action-button" onClick={capture}>Capturar Foto</button>
                    </div>
                )}
                
                {preview && (
                    <div className="preview-container">
                        <h4>Pré-visualização</h4>
                        <img src={preview} alt="Preview" className="preview-image" />
                    </div>
                )}

                <button type="submit" className="action-button">Cadastrar</button>
            </form>
            {message.text && <div className={`message ${message.type}`}>{message.text}</div>}
        </div>
    );
};

export default Register;