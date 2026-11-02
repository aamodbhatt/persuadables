-- V004: deterministic, stratified train/test split.
-- Within each (arm, visit, conversion) stratum, rows are shuffled by a seeded md5 hash and dealt
-- into ten folds; the first test_tenths folds are the test set. md5 (not hash()) keeps the split
-- identical across DuckDB versions and machines.
CREATE OR REPLACE TABLE experiment AS
SELECT
  f.*,
  (
    (
      row_number() OVER (
        PARTITION BY f.arm, f.visit, f.conversion
        ORDER BY md5(concat(f.customer_id, ':', getvariable('seed')))
      ) - 1
    ) % 10
  ) < getvariable('test_tenths') AS is_test
FROM features AS f;
