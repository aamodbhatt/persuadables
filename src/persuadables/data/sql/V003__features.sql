-- V003: model inputs and pre-registered segments.
-- Every column here is measured before the campaign (the 12 months before randomization), so
-- none of them can be affected by the email. Segment definitions are fixed in this file before
-- any effect is estimated; changing them is a new migration, visible in the history.
CREATE OR REPLACE TABLE features AS
SELECT
  customer_id,
  arm,
  visit,
  conversion,
  spend,
  -- model inputs
  recency,
  history,
  ln(1 + history) AS log_history,
  mens,
  womens,
  newbie,
  CAST(zip_area = 'Urban' AS INTEGER) AS zip_urban,
  CAST(zip_area = 'Suburban' AS INTEGER) AS zip_suburban,
  CAST(zip_area = 'Rural' AS INTEGER) AS zip_rural,
  CAST(channel = 'Phone' AS INTEGER) AS channel_phone,
  CAST(channel = 'Web' AS INTEGER) AS channel_web,
  CAST(channel = 'Multichannel' AS INTEGER) AS channel_multichannel,
  -- pre-registered segments
  CASE
    WHEN mens = 1 AND womens = 1 THEN 'both'
    WHEN mens = 1 THEN 'mens only'
    WHEN womens = 1 THEN 'womens only'
    ELSE 'neither'
  END AS purchase_category,
  CASE
    WHEN recency <= 3 THEN '1-3 months'
    WHEN recency <= 6 THEN '4-6 months'
    WHEN recency <= 9 THEN '7-9 months'
    ELSE '10-12 months'
  END AS recency_band,
  CASE
    WHEN history < 200 THEN 'under $200'
    WHEN history < 500 THEN '$200-$500'
    ELSE '$500 and over'
  END AS history_tier,
  channel,
  zip_area,
  CASE newbie WHEN 1 THEN 'new' ELSE 'returning' END AS tenure
FROM customers;
