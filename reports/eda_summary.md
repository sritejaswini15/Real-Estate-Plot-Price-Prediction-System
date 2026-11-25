# Exploratory summary

Source: `99acres_patancheru_properties.xlsx`. 100 raw rows, 94 after cleaning, 81 distinct plots (7 cross-posted).

## Price per sq yd by locality

| Locality | Listings | Median ₹/sq yd | Middle half | Median size (sq yd) |
|---|---:|---:|---:|---:|
| Patighanpur | 7 | 40,000 | 39,750–43,500 | 300 |
| Kistareddypet | 4 | 39,379 | 30,077–52,256 | 349 |
| Isnapur | 21 | 39,228 | 29,902–45,000 | 200 |
| Pati | 4 | 38,000 | 35,975–41,507 | 150 |
| Muthangi | 12 | 35,000 | 29,176–40,417 | 222 |
| Patancheru | 42 | 28,375 | 13,917–39,750 | 200 |
| Other | 4 | 32,475 | 29,902–43,786 | 204 |

## Spearman correlation with log price per sq yd

| Feature | ρ | Listings with value |
|---|---:|---:|
| log plot area | +0.00 | 94 |
| road width (ft) | -0.15 | 61 |
| open sides | +0.02 | 78 |
| in named layout | -0.05 | 94 |

## Dropped rows

- source row 68: plot area 22.11 sq yd outside 50-100000 (likely a unit error)
- 5 exact duplicates: source rows 11, 62, 63, 64, 65

## Figures

- `figures/rate_distribution.png`
- `figures/rate_by_locality.png`
- `figures/price_vs_area.png`
- `figures/rate_by_road_width.png`
- `figures/missing_values.png`
