-- Add a tags column to products for categorization.
-- This migration demonstrates SQLx's embedded migration system.
-- The filename format is: <timestamp>_<description>.sql
-- Migrations run in timestamp order and are tracked in _sqlx_migrations.

ALTER TABLE products ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';

-- Add a comment explaining the column's purpose (good practice for schema changes)
COMMENT ON COLUMN products.tags IS 'Array of string tags for flexible product categorization';
