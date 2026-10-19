# Station data format

No station data are distributed with this repository.

Each station CSV has a header and one row per 15-minute timestamp. Timestamps must be unique, ordered and gap-free.

| Column | Meaning |
|---|---|
| `date` | Timestamp, e.g. `2019-01-01 00:00:00` |
| Five numeric covariate columns | Historical weather measurements; preserve the original feature order |
| `OT` | Raw photovoltaic power, in the same units as `--capacity` |

For daylight evaluation, one covariate name must contain `total solar irradiance` (case-insensitive) and use irradiance units compatible with the fixed threshold of 1. A possible schema is `date,Total solar irradiance,Temperature,Wind speed,Pressure,Diffuse irradiance,OT`; this is an illustrative header, not a claim about the original station feature names.

The loader standardizes the covariates with training-only statistics and places power divided by installed capacity in the last history channel. Do not supply already-standardized power or capacity factors as raw `OT`. Extra columns such as humidity would change the input channel count; remove them explicitly if they are not part of the five-feature protocol. `--capacity` must be supplied for a custom station whose capacity differs from the selected station default.

Names for the experiment grid are `solar_station_1.csv` through `solar_station_4.csv`. The training program records each input file hash and its actual covariate order. Raw station CSV files are excluded by `.gitignore` under this directory.
