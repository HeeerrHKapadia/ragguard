-- Extensions must exist before any table that uses their types.
-- Runs once, on first init of an empty data volume.

-- Vector similarity search: the `vector` column type + HNSW/IVFFlat indexes.
CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram matching. Useful later for fuzzy lexical matching on entity names
-- (people, teams, policy titles) where full-text search is too rigid.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Strips accents so "resume" matches "résumé" in lexical search.
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Note: gen_random_uuid() is built into Postgres 13+, so no uuid-ossp needed.
