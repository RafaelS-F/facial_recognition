import React, { useState } from 'react';
import './App.css';
import Register from './components/Register';
import Verify from './components/Verify';

function App() {
  const [activeTab, setActiveTab] = useState('verify');

  return (
    <div className="App">
      <header className="header">
        <h1>Sistema de Verificação Aeroportuária</h1>
      </header>
      
      <nav className="nav">
        <button 
          className={`nav-button ${activeTab === 'verify' ? 'active' : ''}`} 
          onClick={() => setActiveTab('verify')}
        >
          Verificar Identidade
        </button>
        <button 
          className={`nav-button ${activeTab === 'register' ? 'active' : ''}`}
          onClick={() => setActiveTab('register')}
        >
          Cadastrar Passageiro
        </button>
      </nav>

      <main className="content">
        {activeTab === 'verify' ? <Verify /> : <Register />}
      </main>
    </div>
  );
}

export default App;