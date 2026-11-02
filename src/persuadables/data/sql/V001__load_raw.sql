-- V001: load the raw MineThatData CSV with an explicit schema.
-- auto_detect is off on purpose: a value that does not parse as its declared type is an error
-- here, not a silently widened column further down the pipeline.
CREATE OR REPLACE TABLE raw AS
SELECT *
FROM read_csv(
  getvariable('raw_path'),
  header = true,
  auto_detect = false,
  delim = ',',
  columns = {
    'recency': 'INTEGER',
    'history_segment': 'VARCHAR',
    'history': 'DOUBLE',
    'mens': 'INTEGER',
    'womens': 'INTEGER',
    'zip_code': 'VARCHAR',
    'newbie': 'INTEGER',
    'channel': 'VARCHAR',
    'segment': 'VARCHAR',
    'visit': 'INTEGER',
    'conversion': 'INTEGER',
    'spend': 'DOUBLE'
  }
);
