"""
Test web interface (Streamlit) with real Star Wars data.

This test verifies the complete web interface can render and handle
user interactions using the actual Star Wars dialogue dataset.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest

# We'll import the Streamlit app once it's created
# from src.star_wars_rag.web_app import main


class TestWebInterface:
    """Test Streamlit web interface with real Star Wars data."""
    
    @pytest.fixture(scope="class")
    def real_data_setup(self):
        """Setup real Star Wars data for web interface testing."""
        from src.star_wars_rag import StarWarsChatApp
        
        data_dir = Path("data/raw")
        if not data_dir.exists():
            pytest.skip("Real data directory not found")
        
        script_files = list(data_dir.glob("*.txt"))
        if not script_files:
            pytest.skip("No script files found in data/raw")
        
        # Use A New Hope for testing
        test_script = None
        for script in script_files:
            if "NEW HOPE" in script.name.upper():
                test_script = script
                break
        
        if test_script is None:
            test_script = script_files[0]  # Use first available
        
        # Load and process the data
        app = StarWarsChatApp(auto_download=False)  # Don't auto-download model for testing
        
        # Create temp directory for processing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script_dir = Path(temp_dir) / "scripts"
            temp_script_dir.mkdir()
            
            temp_script = temp_script_dir / test_script.name
            temp_script.write_text(test_script.read_text(encoding='utf-8'), encoding='utf-8')
            
            app.load_from_scripts(temp_script_dir, pattern=test_script.name)
        
        return {
            "app": app,
            "script_name": test_script.name,
            "available_characters": app.get_available_characters()
        }
    
    @pytest.fixture
    def streamlit_app(self, real_data_setup):
        """Create Streamlit app test instance."""
        try:
            from src.star_wars_rag.web_app import main
        except ImportError:
            pytest.skip("Streamlit web app not yet implemented")
        
        # Mock the chat app initialization to use our test data
        with patch('src.star_wars_rag.web_app.initialize_chat_app') as mock_init:
            mock_init.return_value = real_data_setup["app"]
            
            # Create Streamlit test app
            app_test = AppTest.from_function(main)
            return app_test
    
    def test_app_initialization(self, streamlit_app):
        """Test that the Streamlit app initializes correctly."""
        # Run the app
        streamlit_app.run()
        
        # Check that the app loaded without errors
        assert not streamlit_app.exception, f"App initialization failed: {streamlit_app.exception}"
        
        # Check for basic UI elements
        assert len(streamlit_app.title) > 0, "App should have a title"
        assert "Star Wars" in str(streamlit_app.title[0].value), "Title should mention Star Wars"
    
    def test_character_selection(self, streamlit_app, real_data_setup):
        """Test character selection dropdown."""
        streamlit_app.run()
        
        # Should have a character selectbox
        selectboxes = streamlit_app.selectbox
        assert len(selectboxes) > 0, "Should have character selection dropdown"
        
        character_selectbox = None
        for selectbox in selectboxes:
            if "character" in str(selectbox.label).lower():
                character_selectbox = selectbox
                break
        
        assert character_selectbox is not None, "Should have character selection dropdown"
        
        # Check that available characters are in the options
        available_chars = real_data_setup["available_characters"]
        selectbox_options = character_selectbox.options
        
        # At least some main characters should be available
        main_characters = ["Luke Skywalker", "Darth Vader", "Han Solo", "Princess Leia"]
        found_main_chars = [char for char in main_characters if char in selectbox_options]
        
        assert len(found_main_chars) > 0, f"No main characters found in options: {selectbox_options}"
    
    def test_chat_input_interface(self, streamlit_app):
        """Test chat input interface elements."""
        streamlit_app.run()
        
        # Should have text input for user messages
        text_inputs = streamlit_app.text_input
        chat_input = None
        
        for text_input in text_inputs:
            if any(word in str(text_input.label).lower() for word in ["message", "chat", "ask"]):
                chat_input = text_input
                break
        
        assert chat_input is not None, "Should have chat input field"
        
        # Should have send button or similar
        buttons = streamlit_app.button
        send_button = None
        
        for button in buttons:
            if any(word in str(button.label).lower() for word in ["send", "chat", "ask"]):
                send_button = button
                break
        
        assert send_button is not None, "Should have send/chat button"
    
    def test_chat_interaction_with_luke(self, streamlit_app, real_data_setup):
        """Test chat interaction with Luke Skywalker."""
        available_chars = real_data_setup["available_characters"]
        
        if "Luke Skywalker" not in available_chars:
            pytest.skip("Luke Skywalker not available in test data")
        
        streamlit_app.run()
        
        # Select Luke Skywalker
        character_selectbox = None
        for selectbox in streamlit_app.selectbox:
            if "character" in str(selectbox.label).lower():
                character_selectbox = selectbox
                break
        
        if character_selectbox:
            character_selectbox.select("Luke Skywalker")
        
        # Enter a message
        chat_input = None
        for text_input in streamlit_app.text_input:
            if any(word in str(text_input.label).lower() for word in ["message", "chat", "ask"]):
                chat_input = text_input
                break
        
        if chat_input:
            chat_input.input("Tell me about the Force")
        
        # Click send button
        send_button = None
        for button in streamlit_app.button:
            if any(word in str(button.label).lower() for word in ["send", "chat", "ask"]):
                send_button = button
                break
        
        if send_button:
            send_button.click()
        
        # Re-run app to process the interaction
        streamlit_app.run()
        
        # Check for response in the interface
        # This might be in markdown, text, or other elements
        response_found = False
        
        # Check markdown elements for response
        for markdown in streamlit_app.markdown:
            if len(str(markdown.value)) > 10:  # Substantial content
                response_found = True
                break
        
        # Check text elements for response
        if not response_found:
            for text_element in getattr(streamlit_app, 'text', []):
                if len(str(text_element.value)) > 10:
                    response_found = True
                    break
        
        assert response_found, "Should display chat response from Luke"
    
    def test_character_switching(self, streamlit_app, real_data_setup):
        """Test switching between different characters."""
        available_chars = real_data_setup["available_characters"]
        
        if len(available_chars) < 2:
            pytest.skip("Need at least 2 characters for switching test")
        
        streamlit_app.run()
        
        # Get character selectbox
        character_selectbox = None
        for selectbox in streamlit_app.selectbox:
            if "character" in str(selectbox.label).lower():
                character_selectbox = selectbox
                break
        
        if character_selectbox:
            # Switch to first character
            first_char = available_chars[0]
            character_selectbox.select(first_char)
            streamlit_app.run()
            
            # Switch to second character  
            second_char = available_chars[1]
            character_selectbox.select(second_char)
            streamlit_app.run()
            
            # Verify the selection took effect
            current_selection = character_selectbox.value
            assert current_selection == second_char, f"Character switch failed: {current_selection} != {second_char}"
    
    def test_chat_history_display(self, streamlit_app, real_data_setup):
        """Test that chat history is properly displayed."""
        available_chars = real_data_setup["available_characters"]
        
        if len(available_chars) == 0:
            pytest.skip("No characters available for chat history test")
        
        streamlit_app.run()
        
        # Select a character
        character_selectbox = None
        for selectbox in streamlit_app.selectbox:
            if "character" in str(selectbox.label).lower():
                character_selectbox = selectbox
                break
        
        if character_selectbox:
            character_selectbox.select(available_chars[0])
        
        # Send multiple messages to build history
        chat_input = None
        for text_input in streamlit_app.text_input:
            if any(word in str(text_input.label).lower() for word in ["message", "chat", "ask"]):
                chat_input = text_input
                break
        
        send_button = None
        for button in streamlit_app.button:
            if any(word in str(button.label).lower() for word in ["send", "chat", "ask"]):
                send_button = button
                break
        
        if chat_input and send_button:
            # First message
            chat_input.input("Hello")
            send_button.click()
            streamlit_app.run()
            
            # Second message
            chat_input.input("How are you?")
            send_button.click()
            streamlit_app.run()
            
            # Check that multiple messages are displayed
            message_count = 0
            
            # Count substantial content in markdown/text elements
            for markdown in streamlit_app.markdown:
                content = str(markdown.value)
                if len(content) > 5 and any(word in content.lower() for word in ["hello", "how", "you"]):
                    message_count += 1
            
            assert message_count >= 2, f"Should display multiple messages in history, found {message_count}"
    
    def test_system_stats_display(self, streamlit_app, real_data_setup):
        """Test display of system statistics."""
        streamlit_app.run()
        
        # Look for system stats in the interface
        stats_found = False
        
        # Check sidebar or main content for stats
        for markdown in streamlit_app.markdown:
            content = str(markdown.value).lower()
            if any(word in content for word in ["characters", "lines", "dialogue", "stats"]):
                stats_found = True
                break
        
        # Also check for metrics
        if hasattr(streamlit_app, 'metric'):
            for metric in streamlit_app.metric:
                if metric.label:
                    stats_found = True
                    break
        
        assert stats_found, "Should display system statistics"
    
    def test_error_handling(self, streamlit_app):
        """Test error handling in the web interface."""
        streamlit_app.run()
        
        # Try to send empty message
        chat_input = None
        for text_input in streamlit_app.text_input:
            if any(word in str(text_input.label).lower() for word in ["message", "chat", "ask"]):
                chat_input = text_input
                break
        
        send_button = None
        for button in streamlit_app.button:
            if any(word in str(button.label).lower() for word in ["send", "chat", "ask"]):
                send_button = button
                break
        
        if chat_input and send_button:
            chat_input.input("")  # Empty message
            send_button.click()
            streamlit_app.run()
            
            # Should handle empty message gracefully (no crash)
            assert not streamlit_app.exception, "App should handle empty messages gracefully"
    
    def test_responsive_design_elements(self, streamlit_app):
        """Test that the interface has proper responsive design elements."""
        streamlit_app.run()
        
        # Check for proper layout structure
        has_columns = len(streamlit_app.columns) > 0
        has_sidebar = len(streamlit_app.sidebar) > 0
        has_containers = len(getattr(streamlit_app, 'container', [])) > 0
        
        # Should have some form of organized layout
        assert has_columns or has_sidebar or has_containers, \
            "Should have organized layout (columns, sidebar, or containers)"
    
    def test_real_time_features(self, streamlit_app, real_data_setup):
        """Test real-time features like auto-refresh or streaming."""
        available_chars = real_data_setup["available_characters"]
        
        if len(available_chars) == 0:
            pytest.skip("No characters available for real-time test")
        
        streamlit_app.run()
        
        # Check for real-time elements like auto-refresh
        # This might be implemented with st.rerun, timers, or other mechanisms
        
        # Look for any time-based or refresh mechanisms in the interface
        real_time_features = False
        
        # Check for progress bars (might indicate streaming)
        if hasattr(streamlit_app, 'progress'):
            real_time_features = len(streamlit_app.progress) > 0
        
        # Check for placeholders (might be used for dynamic updates)
        if hasattr(streamlit_app, 'empty'):
            real_time_features = real_time_features or len(streamlit_app.empty) > 0
        
        # For now, just verify the app doesn't crash during rapid interactions
        character_selectbox = None
        for selectbox in streamlit_app.selectbox:
            if "character" in str(selectbox.label).lower():
                character_selectbox = selectbox
                break
        
        if character_selectbox and len(available_chars) > 1:
            # Rapid character switching
            for char in available_chars[:3]:  # Test first 3 characters
                character_selectbox.select(char)
                streamlit_app.run()
                
                # Should handle rapid changes gracefully
                assert not streamlit_app.exception, f"App crashed during character switch to {char}"


@pytest.mark.real_data
@pytest.mark.integration
@pytest.mark.web
class TestWebInterfaceEnd2End:
    """End-to-end web interface tests."""
    
    def test_complete_chat_session(self, real_data_setup):
        """Test a complete chat session from start to finish."""
        try:
            from src.star_wars_rag.web_app import main
        except ImportError:
            pytest.skip("Streamlit web app not yet implemented")
        
        # Mock the initialization
        with patch('src.star_wars_rag.web_app.initialize_chat_app') as mock_init:
            mock_init.return_value = real_data_setup["app"]
            
            app_test = AppTest.from_function(main)
            app_test.run()
            
            # 1. Verify app loads
            assert not app_test.exception, "App should load without errors"
            
            # 2. Select character
            available_chars = real_data_setup["available_characters"]
            if len(available_chars) > 0:
                character_selectbox = None
                for selectbox in app_test.selectbox:
                    if "character" in str(selectbox.label).lower():
                        character_selectbox = selectbox
                        break
                
                if character_selectbox:
                    character_selectbox.select(available_chars[0])
                    app_test.run()
            
            # 3. Send message
            chat_input = None
            for text_input in app_test.text_input:
                if any(word in str(text_input.label).lower() for word in ["message", "chat", "ask"]):
                    chat_input = text_input
                    break
            
            send_button = None
            for button in app_test.button:
                if any(word in str(button.label).lower() for word in ["send", "chat", "ask"]):
                    send_button = button
                    break
            
            if chat_input and send_button:
                chat_input.input("What is your greatest challenge?")
                send_button.click()
                app_test.run()
                
                # 4. Verify response
                assert not app_test.exception, "Chat interaction should not cause errors"
                
                # Should have some form of response display
                response_elements = len(app_test.markdown) + len(getattr(app_test, 'text', []))
                assert response_elements > 0, "Should display chat response"
    
    def test_performance_with_real_data(self, real_data_setup):
        """Test web interface performance with real data."""
        try:
            from src.star_wars_rag.web_app import main
        except ImportError:
            pytest.skip("Streamlit web app not yet implemented")
        
        with patch('src.star_wars_rag.web_app.initialize_chat_app') as mock_init:
            mock_init.return_value = real_data_setup["app"]
            
            start_time = time.time()
            app_test = AppTest.from_function(main)
            app_test.run()
            load_time = time.time() - start_time
            
            # App should load reasonably quickly (< 10 seconds)
            assert load_time < 10.0, f"App took too long to load: {load_time} seconds"
            assert not app_test.exception, "App should load without errors"
