-- V002: one typed row per customer, with the treatment arm as an integer code.
-- customer_id is the 1-based row position in the source file (rowid follows insertion order),
-- so every downstream id maps back to a line of the published CSV.
-- An unknown arm label maps to NULL and is caught by the null check in validation.
CREATE OR REPLACE TABLE customers AS
SELECT
  CAST(rowid + 1 AS INTEGER) AS customer_id,
  CASE segment
    WHEN 'No E-Mail' THEN 0
    WHEN 'Mens E-Mail' THEN 1
    WHEN 'Womens E-Mail' THEN 2
  END AS arm,
  recency,
  history,
  mens,
  womens,
  newbie,
  -- The source spells this level "Surburban".
  CASE zip_code WHEN 'Surburban' THEN 'Suburban' ELSE zip_code END AS zip_area,
  channel,
  visit,
  conversion,
  spend
FROM raw;
