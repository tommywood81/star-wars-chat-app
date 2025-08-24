import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [selectedCharacter, setSelectedCharacter] = useState('Luke Skywalker');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [lastRequestData, setLastRequestData] = useState(null);
  const [lastLLMRequest, setLastLLMRequest] = useState(null);
  const [lastLLMResponse, setLastLLMResponse] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const characters = [
    { name: 'Luke Skywalker', description: 'Jedi Knight' },
    { name: 'Darth Vader', description: 'Sith Lord' },
    { name: 'Yoda', description: 'Jedi Master' },
    { name: 'Han Solo', description: 'Smuggler' },
    { name: 'Princess Leia', description: 'Rebel Leader' },
    { name: 'Obi-Wan Kenobi', description: 'Jedi Master' }
  ];

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob) => {
    setIsProcessing(true);
    
    try {
      // Step 1: Send audio to STT service
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');
      
      const sttResponse = await axios.post(`${process.env.REACT_APP_STT_URL || 'http://localhost:5001'}/transcribe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const transcription = sttResponse.data.text;
      
      // Add user message
      const userMessage = { type: 'user', text: transcription, timestamp: new Date() };
      setMessages(prev => [...prev, userMessage]);
      
      // Prepare LLM request data
      const llmRequestData = {
        message: transcription,
        character: selectedCharacter
      };
      
      // Store the exact request being sent to LLM
      setLastLLMRequest({
        url: `${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        payload: llmRequestData,
        timestamp: new Date().toISOString()
      });
      
      // Step 2: Send text to LLM service
      const llmResponse = await axios.post(`${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`, llmRequestData);
      
      // Store the complete LLM response for RAG explainability
      setLastLLMResponse(llmResponse.data);
      
      const characterResponse = llmResponse.data.response;
      
      // Add character response
      const characterMessage = { type: 'character', text: characterResponse, timestamp: new Date() };
      setMessages(prev => [...prev, characterMessage]);
      
      // Step 3: Convert response to speech
      const ttsResponse = await axios.post(`${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/synthesize`, {
        text: characterResponse,
        voice: 'en'
      });
      
      // Play the audio response
      const audioFilename = ttsResponse.data.audio_file.split('/').pop(); // Extract just the filename
      const audioUrl = `${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/audio/${audioFilename}`;
      const audio = new Audio(audioUrl);
      audio.play();
      
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Error processing your message. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const sendTextMessage = async () => {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    if (!text) return;
    
    textInput.value = '';
    
    // Add user message
    const userMessage = { type: 'user', text, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    
    setIsProcessing(true);
    
    try {
      // Prepare request data for explainability
      const requestData = {
        message: text,
        character: selectedCharacter,
        timestamp: new Date().toISOString()
      };
      
      // Store the exact request being sent to LLM
      setLastLLMRequest({
        url: `${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        payload: requestData,
        timestamp: new Date().toISOString()
      });
      
      // Send text to LLM service
      const llmResponse = await axios.post(`${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`, requestData);
      
      // Store request and response data for explainability
      setLastRequestData({
        ...requestData,
        response: llmResponse.data.response,
        llmResponse: llmResponse.data
      });
      
      // Store the complete LLM response for RAG explainability
      setLastLLMResponse(llmResponse.data);
      
      const characterResponse = llmResponse.data.response;
      
      // Add character response
      const characterMessage = { type: 'character', text: characterResponse, timestamp: new Date() };
      setMessages(prev => [...prev, characterMessage]);
      
      // Convert response to speech
      const ttsResponse = await axios.post(`${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/synthesize`, {
        text: characterResponse,
        voice: 'en'
      });
      
      // Play the audio response
      const audioFilename = ttsResponse.data.audio_file.split('/').pop(); // Extract just the filename
      const audioUrl = `${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/audio/${audioFilename}`;
      const audio = new Audio(audioUrl);
      audio.play();
      
    } catch (error) {
      console.error('Error sending message:', error);
      alert('Error sending your message. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">Star Wars Chat</h1>
        <p className="subtitle">Chat with your favorite Star Wars characters</p>
        <button 
          className="explainability-button"
          onClick={() => setShowExplainability(true)}
        >
          🔍 Explainability
        </button>
      </header>

      <div className="chat-container">
        <div className="character-selector">
          <h3>Choose Your Character</h3>
          <div className="character-grid">
            {characters.map(character => (
              <div
                key={character.name}
                className={`character-card ${selectedCharacter === character.name ? 'selected' : ''}`}
                onClick={() => setSelectedCharacter(character.name)}
              >
                <h4>{character.name}</h4>
                <p>{character.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-content">
                <strong>{message.type === 'user' ? 'You' : selectedCharacter}:</strong>
                <p>{message.text}</p>
                <small>{message.timestamp.toLocaleTimeString()}</small>
              </div>
            </div>
          ))}
          {isProcessing && (
            <div className="message processing">
              <div className="message-content">
                <p>Processing...</p>
              </div>
            </div>
          )}
        </div>

        <div className="input-section">
          <div className="text-input-container">
            <input
              id="text-input"
              type="text"
              placeholder="Type your message..."
              onKeyPress={(e) => e.key === 'Enter' && sendTextMessage()}
              disabled={isProcessing}
            />
            <button onClick={sendTextMessage} disabled={isProcessing}>
              Send
            </button>
          </div>
          
          <div className="voice-controls">
            <button
              className={`record-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>
          </div>
        </div>
      </div>

      {/* Explainability Modal */}
      {showExplainability && (
        <div className="modal-overlay" onClick={() => setShowExplainability(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>🔍 System Explainability</h2>
              <button 
                className="close-button"
                onClick={() => setShowExplainability(false)}
              >
                ×
              </button>
            </div>
            
                         <div className="modal-body">
               <div className="explainability-section">
                 {lastLLMRequest && (
                   <div className="llm-request-section">
                     <h3>📤 Last Request Sent to LLM Service</h3>
                     <div className="code-block">
                       <h4>Request URL:</h4>
                       <pre>{lastLLMRequest.url}</pre>
                     </div>
                     <div className="code-block">
                       <h4>Request Method:</h4>
                       <pre>{lastLLMRequest.method}</pre>
                     </div>
                     <div className="code-block">
                       <h4>Request Headers:</h4>
                       <pre>{JSON.stringify(lastLLMRequest.headers, null, 2)}</pre>
                     </div>
                     <div className="code-block">
                       <h4>Request Payload (What gets sent to LLM):</h4>
                       <pre>{JSON.stringify(lastLLMRequest.payload, null, 2)}</pre>
                     </div>
                     <div className="code-block">
                       <h4>Request Timestamp:</h4>
                       <pre>{lastLLMRequest.timestamp}</pre>
                     </div>
                   </div>
                 )}

                 {lastLLMResponse && lastLLMResponse.complete_prompt && (
                   <div className="complete-prompt-section">
                     <h3>📝 Complete Prompt Sent to LLM</h3>
                     <p>This is the exact prompt that was sent to the Phi-2 model, including character context, RAG context, and your question:</p>
                     <div className="code-block">
                       <h4>Full Prompt:</h4>
                       <pre>{lastLLMResponse.complete_prompt}</pre>
                     </div>
                     <div className="prompt-metadata">
                       <p><strong>Prompt Length:</strong> {lastLLMResponse.metadata?.prompt_length || 0} characters</p>
                       <p><strong>RAG Enabled:</strong> {lastLLMResponse.metadata?.rag_enabled ? 'Yes' : 'No'}</p>
                     </div>
                   </div>
                 )}

                 {lastLLMResponse && lastLLMResponse.rag_context && lastLLMResponse.rag_context.length > 0 && (
                   <div className="rag-context-section">
                     <h3>🎬 RAG Context - Movie Lines Retrieved</h3>
                     <p>These are the exact Star Wars dialogue lines that were retrieved and included in the prompt above:</p>
                     <div className="rag-context-list">
                       {lastLLMResponse.rag_context.map((context, index) => (
                         <div key={index} className="rag-context-item">
                           <div className="rag-context-header">
                             <span className="movie-title">{context.movie_title}</span>
                             <span className="scene-info">{context.scene_info}</span>
                           </div>
                           <div className="rag-context-dialogue">
                             <strong>Dialogue:</strong> "{context.dialogue}"
                           </div>
                           {context.cleaned_dialogue && context.cleaned_dialogue !== context.dialogue && (
                             <div className="rag-context-cleaned">
                               <strong>Cleaned:</strong> "{context.cleaned_dialogue}"
                             </div>
                           )}
                         </div>
                       ))}
                     </div>
                     <div className="rag-metadata">
                       <p><strong>Total Context Lines:</strong> {lastLLMResponse.rag_context.length}</p>
                     </div>
                   </div>
                 )}

                 <h3>🤖 How It Works</h3>
                 <p>This Star Wars Chat uses a sophisticated AI pipeline with multiple specialized services:</p>
                
                <div className="service-explanation">
                  <h4>🎤 Speech-to-Text (STT)</h4>
                  <p>Uses OpenAI's Whisper base model to convert your voice to text. The model is optimized for speed and accuracy.</p>
                  
                  <h4>🧠 Large Language Model (LLM)</h4>
                  <p>Uses Microsoft's Phi-2 model with RAG (Retrieval-Augmented Generation) to generate responses. The system:</p>
                  <ul>
                    <li>Searches a vector database of Star Wars dialogue</li>
                    <li>Finds the most relevant context from the original trilogy</li>
                    <li>Generates responses in the character's style</li>
                  </ul>
                  
                  <h4>🔊 Text-to-Speech (TTS)</h4>
                  <p>Uses gTTS (Google Text-to-Speech) to convert the AI response back to speech.</p>
                </div>

                {lastRequestData && (
                  <div className="request-data">
                    <h3>📤 Last Request to LLM</h3>
                    <div className="code-block">
                      <h4>Request Payload:</h4>
                      <pre>{JSON.stringify(lastRequestData, null, 2)}</pre>
                    </div>
                    
                    {lastRequestData.llmResponse && (
                      <div className="code-block">
                        <h4>LLM Response Data:</h4>
                        <pre>{JSON.stringify(lastRequestData.llmResponse, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                )}

                <div className="technical-details">
                  <h3>⚙️ Technical Details</h3>
                  <ul>
                    <li><strong>STT Model:</strong> Whisper Base (~74MB) - CPU optimized</li>
                    <li><strong>LLM Model:</strong> Microsoft Phi-2 (2.7B parameters)</li>
                    <li><strong>Vector Database:</strong> PostgreSQL with pgvector</li>
                    <li><strong>TTS Engine:</strong> Google Text-to-Speech</li>
                    <li><strong>Architecture:</strong> Microservices with Docker containers</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
