-- Initialize Star Wars RAG Database

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Movies table
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    year INTEGER,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Characters table  
CREATE TABLE IF NOT EXISTS characters (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dialogue lines table with vector embeddings
CREATE TABLE IF NOT EXISTS dialogue_lines (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(id),
    character_id INTEGER REFERENCES characters(id),
    dialogue TEXT NOT NULL,
    cleaned_dialogue TEXT,
    scene_info TEXT,
    line_number INTEGER,
    word_count INTEGER,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_embedding ON dialogue_lines USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_character ON dialogue_lines(character_id);
CREATE INDEX IF NOT EXISTS idx_dialogue_lines_movie ON dialogue_lines(movie_id);

-- Insert basic movies
INSERT INTO movies (title, year, description) VALUES 
    ('Star Wars: Episode IV - A New Hope', 1977, 'The original Star Wars film'),
    ('Star Wars: Episode V - The Empire Strikes Back', 1980, 'The second Star Wars film'),
    ('Star Wars: Episode VI - Return of the Jedi', 1983, 'The third Star Wars film')
ON CONFLICT (title) DO NOTHING;

-- Insert basic characters
INSERT INTO characters (name) VALUES 
    ('Luke Skywalker'),
    ('Darth Vader'),
    ('Yoda'),
    ('Han Solo'),
    ('Princess Leia'),
    ('Obi-Wan Kenobi'),
    ('C-3PO'),
    ('R2-D2'),
    ('Chewbacca'),
    ('Lando Calrissian')
ON CONFLICT (name) DO NOTHING;

-- Insert sample dialogue lines
INSERT INTO dialogue_lines (movie_id, character_id, dialogue, cleaned_dialogue, word_count, embedding) 
SELECT 
    m.id as movie_id,
    c.id as character_id,
    d.dialogue,
    d.dialogue as cleaned_dialogue,
    array_length(string_to_array(d.dialogue, ' '), 1) as word_count,
    array_fill(0.0::float, ARRAY[384]) as embedding
FROM (VALUES 
    ('Star Wars: Episode IV - A New Hope', 'Luke Skywalker', 'The Force is strong with this one.'),
    ('Star Wars: Episode IV - A New Hope', 'Darth Vader', 'I find your lack of faith disturbing.'),
    ('Star Wars: Episode V - The Empire Strikes Back', 'Yoda', 'Do or do not. There is no try.'),
    ('Star Wars: Episode IV - A New Hope', 'Han Solo', 'May the Force be with you.'),
    ('Star Wars: Episode IV - A New Hope', 'Princess Leia', 'Help me, Obi-Wan Kenobi. You''re my only hope.')
) AS d(movie_title, character_name, dialogue)
JOIN movies m ON m.title = d.movie_title
JOIN characters c ON c.name = d.character_name;
