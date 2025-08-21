"""
Test PostgreSQL + pgvector database integration with real Star Wars data.

This test verifies the complete database integration can store and retrieve
embeddings using the actual Star Wars dialogue dataset.
"""

import pytest
import asyncio
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# We'll import database components once they're created
# from src.star_wars_rag.database import DatabaseManager


class TestDatabaseIntegration:
    """Test PostgreSQL + pgvector integration with real Star Wars data."""
    
    @pytest.fixture(scope="class")
    def database_config(self):
        """Database configuration for testing."""
        # Use environment variables or defaults for testing
        return {
            "host": os.getenv("TEST_DB_HOST", "localhost"),
            "port": int(os.getenv("TEST_DB_PORT", "5432")),
            "database": os.getenv("TEST_DB_NAME", "star_wars_test"),
            "user": os.getenv("TEST_DB_USER", "postgres"),
            "password": os.getenv("TEST_DB_PASSWORD", "password")
        }
    
    @pytest.fixture(scope="class")
    def real_data_setup(self):
        """Setup real Star Wars data for database testing."""
        from src.star_wars_rag import StarWarsRAGApp
        
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
        app = StarWarsRAGApp()
        
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
            "dialogue_data": app.dialogue_data,
            "embeddings": app.retriever.embeddings
        }
    
    @pytest.fixture(scope="class")
    async def db_manager(self, database_config):
        """Create database manager with test database."""
        try:
            from src.star_wars_rag.database import DatabaseManager
        except ImportError:
            pytest.skip("Database manager not yet implemented")
        
        db = DatabaseManager(**database_config)
        
        # Initialize database (create tables, extensions)
        await db.initialize()
        
        yield db
        
        # Cleanup: drop test data
        await db.cleanup_test_data()
        await db.close()
    
    @pytest.mark.asyncio
    async def test_database_connection(self, db_manager):
        """Test basic database connection and pgvector extension."""
        # Test connection
        is_connected = await db_manager.test_connection()
        assert is_connected, "Database connection failed"
        
        # Test pgvector extension
        has_pgvector = await db_manager.has_pgvector()
        assert has_pgvector, "pgvector extension not available"
    
    @pytest.mark.asyncio
    async def test_create_tables(self, db_manager):
        """Test creation of required database tables."""
        # Create tables
        await db_manager.create_tables()
        
        # Verify tables exist
        tables = await db_manager.list_tables()
        required_tables = ["dialogue_lines", "characters", "movies", "sessions"]
        
        for table in required_tables:
            assert table in tables, f"Required table '{table}' not found"
    
    @pytest.mark.asyncio
    async def test_store_dialogue_data(self, db_manager, real_data_setup):
        """Test storing real Star Wars dialogue data with embeddings."""
        data = real_data_setup
        dialogue_df = data["dialogue_data"]
        embeddings = data["embeddings"]
        
        # Store dialogue data
        stored_count = await db_manager.store_dialogue_data(dialogue_df, embeddings)
        assert stored_count > 0, "No dialogue data was stored"
        assert stored_count == len(dialogue_df), "Not all dialogue data was stored"
        
        # Verify data was stored correctly
        total_lines = await db_manager.count_dialogue_lines()
        assert total_lines == stored_count, "Stored count doesn't match database count"
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, db_manager, real_data_setup):
        """Test vector similarity search with real embeddings."""
        from src.star_wars_rag import StarWarsEmbedder
        
        data = real_data_setup
        
        # First store the data
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        
        # Create embedder for query
        embedder = StarWarsEmbedder()
        
        # Test similarity search with Force-related query
        query = "Tell me about the Force"
        query_embedding = embedder.embed_text(query)
        
        results = await db_manager.similarity_search(
            query_embedding=query_embedding,
            limit=5
        )
        
        assert len(results) > 0, "No similarity search results found"
        assert len(results) <= 5, "Too many results returned"
        
        # Verify result structure
        for result in results:
            assert "id" in result
            assert "character" in result
            assert "dialogue" in result
            assert "movie" in result
            assert "similarity" in result
            assert isinstance(result["similarity"], (float, int))
            assert 0 <= result["similarity"] <= 1, "Similarity score out of range"
    
    @pytest.mark.asyncio
    async def test_character_filtering(self, db_manager, real_data_setup):
        """Test filtering search results by character."""
        from src.star_wars_rag import StarWarsEmbedder
        
        data = real_data_setup
        
        # Store data
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        
        # Get available characters
        characters = await db_manager.get_characters()
        assert len(characters) > 0, "No characters found in database"
        
        # Test character filtering with Luke Skywalker
        if "Luke Skywalker" in characters:
            test_character = "Luke Skywalker"
        else:
            test_character = characters[0]  # Use first available character
        
        embedder = StarWarsEmbedder()
        query_embedding = embedder.embed_text("What do you think?")
        
        results = await db_manager.similarity_search(
            query_embedding=query_embedding,
            character_filter=test_character,
            limit=3
        )
        
        assert len(results) > 0, f"No results found for character {test_character}"
        
        # All results should be from the specified character
        for result in results:
            assert result["character"] == test_character, \
                f"Result from wrong character: {result['character']} != {test_character}"
    
    @pytest.mark.asyncio
    async def test_movie_filtering(self, db_manager, real_data_setup):
        """Test filtering search results by movie."""
        from src.star_wars_rag import StarWarsEmbedder
        
        data = real_data_setup
        
        # Store data
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        
        # Get available movies
        movies = await db_manager.get_movies()
        assert len(movies) > 0, "No movies found in database"
        
        test_movie = movies[0]  # Use first available movie
        
        embedder = StarWarsEmbedder()
        query_embedding = embedder.embed_text("What's happening?")
        
        results = await db_manager.similarity_search(
            query_embedding=query_embedding,
            movie_filter=test_movie,
            limit=3
        )
        
        assert len(results) > 0, f"No results found for movie {test_movie}"
        
        # All results should be from the specified movie
        for result in results:
            assert result["movie"] == test_movie, \
                f"Result from wrong movie: {result['movie']} != {test_movie}"
    
    @pytest.mark.asyncio
    async def test_session_management(self, db_manager):
        """Test chat session storage and retrieval."""
        session_id = "test-session-123"
        character = "Luke Skywalker"
        
        # Store session messages
        messages = [
            {"role": "user", "content": "Hello Luke"},
            {"role": "character", "content": "Hello there!"},
            {"role": "user", "content": "Tell me about the Force"},
            {"role": "character", "content": "The Force is what gives a Jedi his power..."}
        ]
        
        for message in messages:
            await db_manager.store_session_message(
                session_id=session_id,
                character=character,
                role=message["role"],
                content=message["content"]
            )
        
        # Retrieve session history
        history = await db_manager.get_session_history(session_id, limit=10)
        
        assert len(history) == len(messages), "Not all messages were stored/retrieved"
        
        # Verify message order and content
        for i, (stored_msg, original_msg) in enumerate(zip(history, messages)):
            assert stored_msg["role"] == original_msg["role"]
            assert stored_msg["content"] == original_msg["content"]
            assert stored_msg["character"] == character
    
    @pytest.mark.asyncio
    async def test_character_statistics(self, db_manager, real_data_setup):
        """Test character dialogue statistics."""
        data = real_data_setup
        
        # Store data
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        
        # Get character stats
        stats = await db_manager.get_character_statistics()
        
        assert len(stats) > 0, "No character statistics found"
        
        # Verify structure
        for char_stat in stats:
            assert "character" in char_stat
            assert "dialogue_count" in char_stat
            assert "total_words" in char_stat
            assert char_stat["dialogue_count"] > 0
            assert char_stat["total_words"] > 0
        
        # Top character should have substantial dialogue
        top_character = max(stats, key=lambda x: x["dialogue_count"])
        assert top_character["dialogue_count"] > 10, "Top character should have many lines"
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self, db_manager, real_data_setup):
        """Test that stored embeddings maintain correct dimensions."""
        data = real_data_setup
        embeddings = data["embeddings"]
        
        # Store data
        await db_manager.store_dialogue_data(data["dialogue_data"], embeddings)
        
        # Retrieve a sample embedding
        sample_result = await db_manager.similarity_search(
            query_embedding=embeddings[0],  # Use first embedding as query
            limit=1
        )
        
        assert len(sample_result) > 0, "No results from similarity search"
        
        # Check if we can get the raw embedding
        embedding_dim = await db_manager.get_embedding_dimension()
        assert embedding_dim == embeddings.shape[1], \
            f"Embedding dimension mismatch: {embedding_dim} != {embeddings.shape[1]}"
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, db_manager, real_data_setup):
        """Test concurrent database operations."""
        import asyncio
        from src.star_wars_rag import StarWarsEmbedder
        
        data = real_data_setup
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        
        embedder = StarWarsEmbedder()
        
        async def search_task(query_text, task_id):
            query_embedding = embedder.embed_text(f"{query_text} {task_id}")
            results = await db_manager.similarity_search(
                query_embedding=query_embedding,
                limit=3
            )
            return len(results)
        
        # Run multiple concurrent searches
        tasks = [
            search_task("What is the Force?", i)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All searches should return results
        for result_count in results:
            assert result_count > 0, "Concurrent search returned no results"
    
    @pytest.mark.asyncio
    async def test_database_performance(self, db_manager, real_data_setup):
        """Test database performance with real data."""
        import time
        from src.star_wars_rag import StarWarsEmbedder
        
        data = real_data_setup
        
        # Measure storage time
        start_time = time.time()
        await db_manager.store_dialogue_data(data["dialogue_data"], data["embeddings"])
        storage_time = time.time() - start_time
        
        # Storage should be reasonably fast (< 30 seconds for test data)
        assert storage_time < 30, f"Storage took too long: {storage_time} seconds"
        
        # Measure search time
        embedder = StarWarsEmbedder()
        query_embedding = embedder.embed_text("Tell me about the Force")
        
        start_time = time.time()
        results = await db_manager.similarity_search(
            query_embedding=query_embedding,
            limit=10
        )
        search_time = time.time() - start_time
        
        # Search should be fast (< 1 second)
        assert search_time < 1.0, f"Search took too long: {search_time} seconds"
        assert len(results) > 0, "Performance test search returned no results"


@pytest.mark.real_data
@pytest.mark.integration  
@pytest.mark.database
class TestDatabaseIntegrationEnd2End:
    """End-to-end database integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_database_workflow(self):
        """Test complete database workflow from data loading to search."""
        try:
            from src.star_wars_rag.database import DatabaseManager
            from src.star_wars_rag import StarWarsRAGApp, StarWarsEmbedder
        except ImportError:
            pytest.skip("Database components not yet implemented")
        
        # Setup
        db_config = {
            "host": os.getenv("TEST_DB_HOST", "localhost"),
            "port": int(os.getenv("TEST_DB_PORT", "5432")),
            "database": os.getenv("TEST_DB_NAME", "star_wars_test"),
            "user": os.getenv("TEST_DB_USER", "postgres"),
            "password": os.getenv("TEST_DB_PASSWORD", "password")
        }
        
        db = DatabaseManager(**db_config)
        
        try:
            # 1. Initialize database
            await db.initialize()
            await db.create_tables()
            
            # 2. Load real data
            app = StarWarsRAGApp()
            data_dir = Path("data/raw")
            
            if data_dir.exists():
                script_files = list(data_dir.glob("*.txt"))
                if script_files:
                    # Load first available script
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_script_dir = Path(temp_dir) / "scripts"
                        temp_script_dir.mkdir()
                        
                        temp_script = temp_script_dir / script_files[0].name
                        temp_script.write_text(
                            script_files[0].read_text(encoding='utf-8'), 
                            encoding='utf-8'
                        )
                        
                        app.load_from_scripts(temp_script_dir, pattern=script_files[0].name)
            
            # 3. Store in database
            stored_count = await db.store_dialogue_data(
                app.dialogue_data, 
                app.retriever.embeddings
            )
            assert stored_count > 0
            
            # 4. Test search functionality
            embedder = StarWarsEmbedder()
            query_embedding = embedder.embed_text("What is your destiny?")
            
            results = await db.similarity_search(
                query_embedding=query_embedding,
                limit=5
            )
            
            assert len(results) > 0
            assert all("similarity" in r for r in results)
            
            # 5. Test character filtering
            characters = await db.get_characters()
            if len(characters) > 0:
                char_results = await db.similarity_search(
                    query_embedding=query_embedding,
                    character_filter=characters[0],
                    limit=3
                )
                assert len(char_results) > 0
                assert all(r["character"] == characters[0] for r in char_results)
            
        finally:
            await db.cleanup_test_data()
            await db.close()
