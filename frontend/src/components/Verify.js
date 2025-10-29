import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import Webcam from 'react-webcam';

const Verify = () => {
    const [documentId, setDocumentId] = useState('');
    const [photo, setPhoto] = useState(null);
    const [preview, setPreview] = useState(null);
    const [useWebcam, setUseWebcam] = useState(false);
    const [result, setResult] = useState(null);
    const [message, setMessage] = useState({ type: '', text: '' });

    const webcamRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setPhoto(file);
            setPreview(URL.createObjectURL(file));
            setUseWebcam(false);
            setResult(null);
        }
    };

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setPreview(imageSrc);
        fetch(imageSrc).then(res => res.blob()).then(blob => {
            const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
            setPhoto(file);
        });
        setUseWebcam(false);
        setResult(null);
    }, [webcamRef]);

    const handleVerification = async (e) => {
        e.preventDefault();
        if (!documentId || !photo) {
            setMessage({ type: 'error', text: 'Por favor, insira o documento e uma foto para verificação.' });
            return;
        }

        const formData = new FormData();
        formData.append('document_id', documentId);
        formData.append('photo', photo);

        try {
            // --- LINHA CORRIGIDA ---
            const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/verify`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
            setMessage({ type: '', text: '' });
        } catch (error) {
            const errorMsg = error.response ? error.response.data.error : 'Erro ao conectar ao servidor.';
            setMessage({ type: 'error', text: `Falha na verificação: ${errorMsg}` });
            setResult(null);
        }
    };

    return (
        <div>
            <h2>Verificar Identidade do Passageiro</h2>
            <form onSubmit={handleVerification}>
                <div className="form-group">
                    <label>Número do Documento</label>
                    <input type="text" className="form-input" value={documentId} onChange={(e) => setDocumentId(e.target.value)} />
                </div>
                
                <div className="form-group">
                    <label>Foto para Comparação</label>
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
                        <h4>Foto para Verificação</h4>
                        <img src={preview} alt="Verification Preview" className="preview-image" />
                    </div>
                )}

                <button type="submit" className="action-button">Verificar</button>
            </form>
            {message.text && <div className={`message ${message.type}`}>{message.text}</div>}
            
            {result && (
                <div className="result-card" style={{borderColor: result.verified ? 'var(--primary-color)' : '#ff0000'}}>
                    <h3>Resultado da Verificação</h3>
                    <p>Status: <span style={{color: result.verified ? '#00ff00' : '#ff0000'}}>{result.verified ? 'VERIFICADO' : 'NÃO VERIFICADO'}</span></p>
                    <p>Nome: <span>{result.passenger_name}</span></p>
                    <p>Documento: <span>{result.document_id}</span></p>
                    <p>Taxa de Similaridade: <span>{result.similarity_percentage}</span></p>
                </div>
            )}
        </div>
    );
};

export default Verify;