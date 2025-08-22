import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [selectedCharacter, setSelectedCharacter] = useState('Luke Skywalker');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
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
      formData.append('audio', audioBlob, 'audio.wav');
      
      const sttResponse = await axios.post(`${process.env.REACT_APP_STT_URL || 'http://localhost:5001'}/transcribe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const transcription = sttResponse.data.text;
      
      // Add user message
      const userMessage = { type: 'user', text: transcription, timestamp: new Date() };
      setMessages(prev => [...prev, userMessage]);
      
      // Step 2: Send text to LLM service
      const llmResponse = await axios.post(`${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`, {
        message: transcription,
        character: selectedCharacter
      });
      
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
      const audioUrl = `${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/audio/${ttsResponse.data.audio_file}`;
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
      // Send text to LLM service
      const llmResponse = await axios.post(`${process.env.REACT_APP_LLM_URL || 'http://localhost:5003'}/chat`, {
        message: text,
        character: selectedCharacter
      });
      
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
      const audioUrl = `${process.env.REACT_APP_TTS_URL || 'http://localhost:5002'}/audio/${ttsResponse.data.audio_file}`;
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
    </div>
  );
}

export default App;
