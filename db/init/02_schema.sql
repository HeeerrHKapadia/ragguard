-- Core schema for a permission-aware RAG system.
--
-- Phase 0a scope: the identity + corpus skeleton, enough to prove the
-- database works end to end. ACL tables and row-level security arrive in
-- Phase 0b/2, once we have something to protect and a metric that proves
-- it isn't protected yet.

-- ---------------------------------------------------------------------
-- Identity
-- ---------------------------------------------------------------------

-- A tenant is a whole separate organization. This is the HARD isolation
-- boundary: a cross-tenant leak is a critical security bug, never a
-- "relevance" problem. Within-tenant role boundaries are softer and get
-- handled separately.
CREATE TABLE IF NOT EXISTS tenants (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug        text        NOT NULL UNIQUE,
    name        text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email         text        NOT NULL,
    display_name  text        NOT NULL,
    title         text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- Emails are unique per tenant, not globally: the same person could
    -- legitimately exist in two tenants.
    UNIQUE (tenant_id, email)
);

-- Groups are how permissions actually get granted in real orgs. Nobody
-- grants doc-by-doc to individuals; they grant to "engineering" or
-- "people-ops". Group membership is where permission bugs hide.
CREATE TABLE IF NOT EXISTS groups (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    slug       text NOT NULL,
    name       text NOT NULL,

    -- Highest sensitivity tier this group can read by default.
    clearance  text NOT NULL DEFAULT 'internal'
               CHECK (clearance IN ('public', 'internal', 'confidential', 'restricted')),

    -- Sections where this group is elevated beyond its clearance. Finance
    -- sits at 'internal' clearance generally, but reads confidential material
    -- inside `finance/`. Real access is rarely a single global level, and
    -- modelling it as one is how systems end up either over-blocking or
    -- over-sharing.
    elevated_sections text[] NOT NULL DEFAULT '{}',

    UNIQUE (tenant_id, slug)
);

CREATE TABLE IF NOT EXISTS user_groups (
    user_id   uuid NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    group_id  uuid NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_id)
);

-- ---------------------------------------------------------------------
-- Corpus
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    uuid        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_uri   text        NOT NULL,
    title        text        NOT NULL,

    -- The handbook department this document came from. Drives both the
    -- sensitivity tier and the per-section elevation checks above, so it
    -- has to be stored rather than re-derived from the path at query time.
    section      text        NOT NULL DEFAULT 'root',

    -- A CHECK constraint rather than a Postgres ENUM: adding a tier later
    -- is a one-line ALTER instead of an enum migration dance.
    sensitivity  text        NOT NULL DEFAULT 'internal'
                 CHECK (sensitivity IN ('public', 'internal', 'confidential', 'restricted')),
    content_hash text,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_uri)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- DENORMALIZED ON PURPOSE. tenant_id is derivable from document_id via
    -- a join, but retrieval must filter by tenant on EVERY query, and a
    -- join inside a filtered vector search is exactly what wrecks recall
    -- and latency. The filter column has to live on the row being scanned.
    -- This duplication is the single most important design choice here.
    tenant_id   uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,

    ordinal     int  NOT NULL,
    text        text NOT NULL,
    token_count int,

    -- 384 dims: BAAI/bge-small-en-v1.5, which fastembed serves via ONNX.
    -- Chosen for speed and a 133MB download so CI can run the real pipeline
    -- rather than a mocked one. Vectors arrive pre-normalized, so cosine
    -- distance reduces to a dot product.
    --
    -- pgvector fixes this width at table-creation time, so switching model
    -- means recreating the volume. Phase 3 benchmarks larger models and has
    -- to pay that cost deliberately — which is the point, since the upgrade
    -- then comes with a measured quality delta instead of an assumption.
    embedding   vector(384),

    -- Generated column: Postgres keeps the full-text vector in sync
    -- automatically. This is the BM25-ish half of hybrid search in Phase 3.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    UNIQUE (document_id, ordinal)
);

-- ---------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS chunks_tenant_idx      ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_document_idx    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx         ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS documents_tenant_idx   ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS documents_sens_idx     ON documents (sensitivity);
CREATE INDEX IF NOT EXISTS documents_section_idx  ON documents (tenant_id, section);

-- DELIBERATELY NOT CREATED YET: the HNSW index on chunks.embedding.
--
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
--
-- Without it, Postgres does exact nearest-neighbour search: perfect recall,
-- fine at small scale. We add it in Phase 1 and measure what approximate
-- search costs us in recall — and in Phase 3, what it costs us again once
-- permission filters are applied on top. Creating it now would hide both
-- lessons behind "it just worked".
