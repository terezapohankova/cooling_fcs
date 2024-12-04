```mermaid
graph TD
  A[Start] --> B[Initialize Variables]
  B -->|Constants| C[Set Constants: Z, MEASURING_HEIGHT, BLENDING_HEIGHT, cp, B_GAMMA, SOLAR_CONSTANT]
  B -->|Paths| D[Set Paths: INPUT_FOLDER, METEOROLOGY, HILLSHADE, VEG_HEIGHT, OUTPUT_FOLDER]
  B --> E[Create Folders]
  B --> F[Load JSON_MTL_PATH and ORIGINAL_IMG]

  F --> G[Parse Sensing Dates]
  G --> H[Load JSON Metadata]

  H --> I[Create Clipped Image Paths]

  I --> J[Build imgDict]
  J --> K[Calculate Meteorological Parameters]
  K -->|supportlib_v2| L[Calculate zenithAngle, inverseSE, emissivity_atmos, slope_vap_press]

  J --> M[Calculate Vegetation Indices]
  M --> N[NDVI]
  M --> O[SAVI]
  M --> P[LAI]

  J --> Q[Calculate Surface Emissivity]
  Q --> R[Emissivity Calculation]

  J --> S[Perform Thermal Corrections]
  S --> T[Sensor Radiance]
  S --> U[Brightness Temperature]
  S --> V[Land Surface Temperature]

  J --> W[Calculate Albedo]
  W --> X[Reflectivity for B2-B7]
  W --> Y[Albedo Calculation]

  J --> Z[Compute Radiation]
  Z --> AA[Longwave Radiation Outwards]
  Z --> AB[Shortwave Radiation Inwards]
  Z --> AC[Net Radiation]

  J --> AD[Perform S-SEBI Calculations]
  AD --> AE[Soil Heat Flux]
  AD --> AF[Latent Heat Flux]
  AD --> AG[Sensible Heat Flux]

  J --> AH[Priestley-Taylor Calculation]
  AH --> AI[Calculate Evapotranspiration]

  J --> AJ[Penman-Monteith Calculation]
  AJ --> AK[Evapotranspiration]

  J --> AL[SSEBOP Calculation]
  AL --> AM[Evaporative Fraction]

  J --> AN[CCI Calculation]
  AN --> AO[ETI and CCI Calculation]

  AO --> AP[End]
  ```