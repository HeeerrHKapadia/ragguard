-- Phase B prep: denormalize ACL fields onto chunks.
--
-- tenant_id is already on chunks so filtered ANN does not join for the hard
-- isolation boundary. section and sensitivity drive the rest of the policy;
-- keeping them only on documents forces every retrieval query to join
-- documents (and usually tenants) before the distance sort.
--
-- Apply after the base schema. Safe to re-run. Index scripts should populate
-- these columns on insert; the UPDATE below backfills existing rows.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS section text,
    ADD COLUMN IF NOT EXISTS sensitivity text
        CHECK (
            sensitivity IS NULL
            OR sensitivity IN ('public', 'internal', 'confidential', 'restricted')
        );

UPDATE chunks c
   SET section = d.section,
       sensitivity = d.sensitivity
  FROM documents d
 WHERE d.id = c.document_id
   AND (c.section IS DISTINCT FROM d.section
        OR c.sensitivity IS DISTINCT FROM d.sensitivity);

CREATE INDEX IF NOT EXISTS chunks_acl_idx
    ON chunks (tenant_id, sensitivity);

CREATE INDEX IF NOT EXISTS chunks_section_idx
    ON chunks (tenant_id, section);
