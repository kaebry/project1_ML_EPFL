# Feature notes

Checked against the [CDC codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
and our training data.

- `SEQNO` and `_PSU` are identical survey IDs. Try removing both.
- `_STATE`, `MARITAL` and `EMPLOY1` are categories. Try one-hot encoding later.
- `SEX`: 1 = male, 2 = female. Can be encoded as 0/1.
- `GENHLTH`: 1-5 are ordered health ratings; 7 and 9 mean unavailable.
- `PHYSHLTH`: 88 means zero days; 77 and 99 mean unavailable.
- `EMPLOY1`: 7 means retired, so keep it as a valid category.
- `_AGE80`: age in years, with 80 meaning 80 or older.
- `_BMI5` already contains decimal BMI values. Keep its current units.

In `train.py`, `use_feature_rules=True` removes the IDs, fixes the health codes
and adds two missing indicators. Set it to `False` for the original preprocessing.
