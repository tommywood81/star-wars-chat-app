# Enhanced Explainability Feature

## Overview

The Star Wars Chat App now includes a comprehensive explainability feature that shows all data sent to and received from the LLM service container. This provides complete transparency into how the RAG (Retrieval-Augmented Generation) system works.

## What the Explainability Panel Shows

When you click the "🔍 Show Explainability" button in the dashboard, you'll see a tabbed interface with the following information:

### 📤 Request Data
- **User message**: The exact text you sent
- **Character selected**: Which Star Wars character you're chatting with
- **Model parameters**: max_tokens, temperature, and other generation settings
- **Context**: Any additional context provided

### 📚 Retrieved Context (Matching Movie Lines)
- **Number of lines**: How many relevant dialogue lines were found
- **For each line**:
  - **Dialogue**: The exact quote from the Star Wars movies
  - **Movie**: Which movie the line is from (Episode IV, V, or VI)
  - **Scene info**: Additional scene context
  - **Cleaned version**: Processed version of the dialogue

### 🤖 Complete Prompt Sent to Phi-2 Model
- **Full prompt**: The exact text sent to the Phi-2 model
- **Character description**: Personality and speaking style information
- **Context integration**: How the retrieved movie lines are incorporated
- **Statistics**: Character count, word count, and line count

### 📊 Model Output
- **Generated response**: The character's reply
- **Response statistics**: Length and structure analysis
- **Quality metrics**: Character count, word count, sentence count

### ⚙️ Technical Details
- **Model information**: Phi-2 model details and configuration
- **Performance metrics**: Processing time, tokens generated, context lines retrieved
- **System metadata**: Database status, RAG configuration, and other technical details

## How It Works

1. **User Input**: You type a message and select a character
2. **Vector Search**: The system finds the 6 most relevant dialogue lines from the Star Wars movies
3. **Context Building**: These lines are formatted into context for the LLM
4. **Prompt Construction**: A complete prompt is built with character info, context, and your message
5. **Model Generation**: The Phi-2 model generates a response
6. **Data Capture**: All intermediate data is captured and stored
7. **Explainability Display**: The dashboard shows all this data in an organized, tabbed interface

## Technical Implementation

### LLM Service Changes
- Modified `_generate_response()` to return both response and prompt
- Enhanced `chat_with_character()` to capture all request and response data
- Updated `ChatResponse` model to include new explainability fields
- Added proper error handling and data validation

### Dashboard Changes
- Updated API endpoint to point to LLM service container
- Enhanced explainability panel with tabbed interface
- Added comprehensive data display with statistics
- Improved error handling and user feedback

### Data Flow
```
User Message → Dashboard → LLM Service → Vector Search → Context Retrieval → Prompt Construction → Model Generation → Response + Metadata → Dashboard → Explainability Display
```

## Benefits

1. **Transparency**: Users can see exactly how the system works
2. **Debugging**: Developers can trace issues through the entire pipeline
3. **Education**: Shows the power of RAG systems in action
4. **Quality Assurance**: Verify that the system is using relevant context
5. **Performance Monitoring**: Track processing times and token usage

## Usage

1. Start the application with Docker Compose
2. Navigate to the dashboard
3. Select a character and send a message
4. Click "🔍 Show Explainability" to see all the data
5. Explore the different tabs to understand the system's operation

## Testing

Run the test script to verify the explainability feature:

```bash
python test_explainability.py
```

This will test all the explainability data fields and confirm they're working correctly.

## Future Enhancements

- Add similarity scores for retrieved context lines
- Include token usage breakdown (prompt vs. completion)
- Add visualizations for the RAG pipeline
- Include confidence scores for model responses
- Add export functionality for explainability data
