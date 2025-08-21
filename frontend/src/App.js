import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiMic, FiMicOff, FiVolume2, FiVolumeX, FiSend } from 'react-icons/fi';
import toast, { Toaster } from 'react-hot-toast';
import axios from 'axios';

// Styled Components
const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  font-family: 'Arial', sans-serif;
`;

const Header = styled(motion.header)`
  text-align: center;
  margin-bottom: 30px;
  color: #fff;
`;

const Title = styled.h1`
  font-size: 3rem;
  margin: 0;
  background: linear-gradient(45deg, #ffd700, #ff6b35);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: #ccc;
  margin: 10px 0 0 0;
`;

const ChatContainer = styled(motion.div)`
  width: 100%;
  max-width: 800px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
`;

const CharacterSelector = styled.div`
  padding: 20px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const CharacterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-top: 15px;
`;

const CharacterCard = styled(motion.div)`
  background: ${props => props.selected ? 'rgba(255, 215, 0, 0.2)' : 'rgba(255, 255, 255, 0.1)'};
  border: 2px solid ${props => props.selected ? '#ffd700' : 'rgba(255, 255, 255, 0.2)'};
  border-radius: 10px;
  padding: 15px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  color: #fff;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
  }
`;

const CharacterName = styled.h3`
  margin: 0 0 5px 0;
  font-size: 1rem;
  color: ${props => props.selected ? '#ffd700' : '#fff'};
`;

const CharacterDesc = styled.p`
  margin: 0;
  font-size: 0.8rem;
  color: #ccc;
`;

const MessagesContainer = styled.div`
  height: 400px;
  overflow-y: auto;
  padding: 20px;
  background: rgba(0, 0, 0, 0.2);
`;

const Message = styled(motion.div)`
  margin-bottom: 15px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
`;

const MessageBubble = styled.div`
  background: ${props => props.isUser ? 'rgba(255, 215, 0, 0.2)' : 'rgba(255, 255, 255, 0.1)'};
  border: 1px solid ${props => props.isUser ? '#ffd700' : 'rgba(255, 255, 255, 0.2)'};
  border-radius: 15px;
  padding: 12px 16px;
  max-width: 70%;
  color: #fff;
  word-wrap: break-word;
`;

const MessageSender = styled.div`
  font-weight: bold;
  margin-bottom: 5px;
  color: ${props => props.isUser ? '#ffd700' : '#fff'};
`;

const InputContainer = styled.div`
  padding: 20px;
  background: rgba(0, 0, 0, 0.3);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  gap: 10px;
  align-items: center;
`;

const Input = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;

  &::placeholder {
    color: #ccc;
  }

  &:focus {
    border-color: #ffd700;
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }
`;

const Button = styled(motion.button)`
  padding: 12px 16px;
  border: none;
  border-radius: 50%;
  background: ${props => props.variant === 'primary' ? '#ffd700' : 'rgba(255, 255, 255, 0.2)'};
  color: ${props => props.variant === 'primary' ? '#000' : '#fff'};
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  transition: all 0.3s ease;

  &:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  font-size: 0.9rem;
  color: #ccc;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.status === 'connected' ? '#4CAF50' : '#f44336'};
`;

// Character data
const characters = [
  {
    name: 'Luke Skywalker',
    description: 'Jedi Knight and hero of the Rebellion',
    voiceStyle: 'young_male'
  },
  {
    name: 'Darth Vader',
    description: 'Dark Lord of the Sith',
    voiceStyle: 'deep_male'
  },
  {
    name: 'Princess Leia',
    description: 'Princess of Alderaan and Rebel leader',
    voiceStyle: 'female'
  },
  {
    name: 'Han Solo',
    description: 'Smuggler and captain of the Millennium Falcon',
    voiceStyle: 'male'
  },
  {
    name: 'Yoda',
    description: 'Wise Jedi Master',
    voiceStyle: 'elderly_male'
  },
  {
    name: 'Obi-Wan Kenobi',
    description: 'Jedi Master and mentor',
    voiceStyle: 'mature_male'
  }
];

function App() {
  const [selectedCharacter, setSelectedCharacter] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [servicesStatus, setServicesStatus] = useState({
    stt: false,
    tts: false,
    llm: false
  });

  // Check service status on component mount
  useEffect(() => {
    checkServicesStatus();
  }, []);

  const checkServicesStatus = async () => {
    const services = ['stt', 'tts', 'llm'];
    const status = {};

    for (const service of services) {
      try {
        const response = await axios.get(`http://localhost:500${service === 'stt' ? '1' : service === 'tts' ? '2' : '3'}/health`);
        status[service] = response.status === 200;
      } catch (error) {
        status[service] = false;
      }
    }

    setServicesStatus(status);
  };

  const handleCharacterSelect = (character) => {
    setSelectedCharacter(character);
    toast.success(`Selected ${character.name} for chat!`);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !selectedCharacter || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      text: userMessage,
      sender: 'You',
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newUserMessage]);

    try {
      // Send message to LLM service
      const response = await axios.post('http://localhost:5003/chat', {
        message: userMessage,
        character: selectedCharacter.name,
        max_tokens: 200,
        temperature: 0.7
      });

      const characterResponse = {
        id: Date.now() + 1,
        text: response.data.response,
        sender: selectedCharacter.name,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, characterResponse]);

      // Convert response to speech
      await convertToSpeech(response.data.response, selectedCharacter.voiceStyle);

    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const convertToSpeech = async (text, voiceStyle) => {
    try {
      const response = await axios.post('http://localhost:5002/synthesize', {
        text: text,
        voice: 'ljspeech', // Default voice for now
        speed: 1.0
      }, {
        responseType: 'blob'
      });

      // Create audio element and play
      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      setIsPlaying(true);
      audio.play();
      
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

    } catch (error) {
      console.error('Error converting to speech:', error);
      toast.error('Failed to convert response to speech.');
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast.error('Speech recording is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsRecording(true);
      
      // Here you would implement the actual recording logic
      // For now, we'll just show a placeholder
      toast.success('Recording started! Click stop to send.');
      
      // Simulate recording for demo purposes
      setTimeout(() => {
        stopRecording();
      }, 5000);

    } catch (error) {
      console.error('Error starting recording:', error);
      toast.error('Failed to start recording.');
    }
  };

  const stopRecording = async () => {
    setIsRecording(false);
    
    // Here you would implement the actual stop recording logic
    // For now, we'll just show a placeholder
    toast.success('Recording stopped! Processing audio...');
    
    // Simulate processing for demo purposes
    setTimeout(() => {
      const transcribedText = "Hello, this is a test transcription.";
      setInputMessage(transcribedText);
      toast.success('Audio transcribed successfully!');
    }, 2000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <AppContainer>
      <Toaster position="top-right" />
      
      <Header
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Title>Star Wars Chat</Title>
        <Subtitle>Talk to your favorite Star Wars characters with voice!</Subtitle>
      </Header>

      <ChatContainer
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <CharacterSelector>
          <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Choose Your Character</h3>
          <CharacterGrid>
            {characters.map((character) => (
              <CharacterCard
                key={character.name}
                selected={selectedCharacter?.name === character.name}
                onClick={() => handleCharacterSelect(character)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <CharacterName selected={selectedCharacter?.name === character.name}>
                  {character.name}
                </CharacterName>
                <CharacterDesc>{character.description}</CharacterDesc>
              </CharacterCard>
            ))}
          </CharacterGrid>
        </CharacterSelector>

        <MessagesContainer>
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', color: '#ccc', marginTop: '50px' }}>
              {selectedCharacter 
                ? `Start chatting with ${selectedCharacter.name}!`
                : 'Select a character to begin chatting!'
              }
            </div>
          ) : (
            messages.map((message) => (
              <Message
                key={message.id}
                initial={{ opacity: 0, x: message.isUser ? 50 : -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MessageBubble isUser={message.isUser}>
                  <MessageSender isUser={message.isUser}>
                    {message.sender}
                  </MessageSender>
                  {message.text}
                </MessageBubble>
              </Message>
            ))
          )}
          {isLoading && (
            <Message>
              <MessageBubble isUser={false}>
                <MessageSender isUser={false}>
                  {selectedCharacter?.name}
                </MessageSender>
                Thinking...
              </MessageBubble>
            </Message>
          )}
        </MessagesContainer>

        <InputContainer>
          <Button
            variant={isRecording ? 'primary' : 'secondary'}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!selectedCharacter}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {isRecording ? <FiMicOff /> : <FiMic />}
          </Button>

          <Input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={selectedCharacter ? `Message ${selectedCharacter.name}...` : 'Select a character first...'}
            disabled={!selectedCharacter || isLoading}
          />

          <Button
            variant="primary"
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || !selectedCharacter || isLoading}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <FiSend />
          </Button>

          <Button
            variant="secondary"
            onClick={() => setIsPlaying(false)}
            disabled={!isPlaying}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {isPlaying ? <FiVolumeX /> : <FiVolume2 />}
          </Button>
        </InputContainer>

        <StatusIndicator>
          <StatusDot status={servicesStatus.stt ? 'connected' : 'disconnected'} />
          <span>STT: {servicesStatus.stt ? 'Connected' : 'Disconnected'}</span>
          <StatusDot status={servicesStatus.tts ? 'connected' : 'disconnected'} />
          <span>TTS: {servicesStatus.tts ? 'Connected' : 'Disconnected'}</span>
          <StatusDot status={servicesStatus.llm ? 'connected' : 'disconnected'} />
          <span>LLM: {servicesStatus.llm ? 'Connected' : 'Disconnected'}</span>
        </StatusIndicator>
      </ChatContainer>
    </AppContainer>
  );
}

export default App;
