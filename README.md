# RotateWise

**A NASA-data crop-rotation explorer for Iowa farmers and extension agents.** A county-scale hindcast that explains
past weather stress. It does not forecast this season.

> RotateWise supports farmers and extension agents. It does not replace farmer judgment or local agricultural
> expertise.

Built Nov 14-15, 2026 for **Veritas AI (Climate Track)**.

**The app is a static website** (`web/`, React + TypeScript, deploys to Vercel): a guided four-step path
(Your county → What hurts yields → Your options → Talk it over) plus an Evidence section for judges and agronomists.
Every number and every strategy-card sentence is precomputed in Python (`src/export_web.py`). The browser only
re-weights the precomputed criteria with the visitor's priorities and looks up precomputed what-if scenarios, so
there is no server, no live model and no external API. The Streamlit app (`app.py`) is kept as an analyst
fallback. This replaces spec decision B2 ("App: Streamlit"), changed with the team's approval on 2026-11-14.

## Problem, user and decision

- **User:** a central-Iowa farmer and the county extension agent they work with.
- **Decision:** which crop rotation to explore further with local experts.
- **Inputs:** a county (demo: Story County, FIPS 19169), the current rotation (corn → soybean) and four priorities
  (conserve water, soil health, income, low labour).
- **Output:** three strategies ranked with a transparent formula. Each has a card listing why it was suggested,
  the evidence, possible benefits, information gaps and risks needing local review. Behind them is an ML model of
  how NASA-observed weather stress hurt corn and soybean yields in past seasons.

**Climate-track decision statement:** the model learns which summer conditions (heat, soil dryness) cut county
yields. The app turns that into the months a field is most exposed, and compares rotations by how much of their
crops' critical growth falls in those months and how long they leave soil bare in storm season.

## Results

**Model.** The pre-committed rule chose **M6, a monotone GAM on 5 features** (July max temperature, Jul-Aug heat
days, Jun-Aug rain, August root-zone wetness, Jun-Aug water balance) for both crops. It beat M0 and M2 in
leave-one-year-out CV, and its 90% bootstrap interval of the RMSE difference against M2 excludes zero.
LightGBM, random forest and elastic net did not do better within one standard error.

Yield-anomaly error in percentage points of trend yield, averaged over years, with 90% year-bootstrap intervals:

| | Corn dev CV 2000-19 | Corn **test 2020-25** | Soy dev CV 2000-19 | Soy **test 2020-25** |
|---|---|---|---|---|
| M0 trend only (anomaly = 0) | 11.53 [9.8, 13.7] | 9.22 [7.8, 10.7] | 11.30 [9.3, 13.6] | 9.06 [8.0, 10.3] |
| M1 persistence | 13.42 | 11.98 | 14.28 | 11.77 |
| M2 3-feature ridge | 11.93 | 9.26 | 11.65 | 8.52 |
| **Selected GAM (RMSE)** | **10.74** [9.3, 12.3] | **8.16** [6.7, 9.7] | **10.71** [9.0, 12.9] | **7.55** [6.3, 9.0] |
| Selected MAE | 8.64 | 6.36 | 9.00 | 6.14 |
| Skill vs M0 (1 - RMSE/RMSE_M0) | +7% | **+12%** | +5% | **+17%** |
| Within-year county ranking (Spearman) | 0.17 | 0.19 | 0.08 | 0.22 |

- The GAM had lower RMSE than M0 in all 6 test years for corn and in 5 of 6 for soybean (2024 tied).
- **2025**, the only test year never seen before the final run: corn predicted +3.1% vs actual +3.3%; soybean
  predicted +4.4% vs actual +5.3%.
- A fixed 2000-2019 trend instead of the expanding trend gives corn 9.08 vs M0 10.01, and soybean 7.68 vs M0 9.18.
- **Honest reading:** the gain over simple baselines is real but small. Within-year county ranking is weak, so
  most of the skill is in the statewide year effect. An expanding-window check on 2008-2019 put the soybean GAM
  slightly *behind* M0 (10.99 vs 10.94).

**Overfitting and robustness (dev years).**

| Check | Corn | Soybean |
|---|---|---|
| Train RMSE vs CV RMSE | 9.81 vs 10.74 | 10.28 vs 10.71 |
| Placebo: weather shuffled across years (3 runs) | 11.70, M0 level ✓ | 11.60, M0 level ✓ |
| Spatial hold-out (5 regions × year) | 10.88 | 11.22 |
| A1 without root-zone wetness | 10.74 (no change) | 10.69 (no change) |
| A2 corn + soybean pooled | 10.74 | 10.51 |
| A3 Theil-Sen trend (different target) | 10.68 | 10.78 |
| A4 features as anomalies from county climatology | 11.21 (worse) | 11.37 (worse) |

POWER's root-zone wetness adds nothing beyond temperature and rain (A1), so the NASA soil-moisture product is
not what carries the model here.

**Uncertainty.** 90% split-conformal intervals are wide: ±18 points (corn) and ±20 points (soybean) of trend yield.
Coverage was 96% (corn) and 99% (soybean) on test years, and 90% on average in dev, but only 60% (corn) and 5% (soy)
in the worst dev year. In a year like 2012, the interval is too narrow. Quantile LightGBM covered only 77-78%.

**Where it fails.**
- **2012 drought** (held out in LOYO): statewide corn predicted −13.4% vs actual −27.0%. The direction is right
  but the loss is half-sized. Story County was predicted −14.8% vs actual −12.1%. The US Drought Monitor had all
  of Story County in severe drought (D2) that summer.
- **2020 derecho** (test year): statewide corn predicted −1.4% vs actual −7.6%; **Story County −3.1% vs −21.1%**.
  Wind damage is not in the features. NOAA SPC logged 216 severe-wind reports in Iowa on 10 Aug 2020 (7 in
  Story County, up to 130 mph). USDA RMA corn indemnities in Story County were **$20.8M for wind vs $1.5M for
  drought** ($339M vs $152M statewide). MODIS August NDVI at Story County fell to 0.69, against 0.80 in 2019 and
  0.75 in 2021. Soybeans, less damaged by wind, were predicted well (−5.8% vs −5.4% statewide).

**Rotations (Story County).** With equal priorities, C (corn → soybean → oats/red clover) ranks first, ahead of
A (current) and B (+ winter rye). The ranking depends on priorities:

| Priorities | Ranking |
|---|---|
| All equal | C > A > B |
| Water and soil high | C > B > A |
| Income and low labour high | A > C > B |
| Soil only | B > C > A |
| 2,000 random priority mixes: share ranked first | C 67%, A 24%, B 8% |

- **Calendar sensitivity:** shifting every crop window 14-21 days *later* moves C from first to last, because the
  oats' critical period then runs into July, the month the model weights most (65% for corn). Shifting earlier
  changes nothing. The ranking is sensitive to timing, which varies by year and field.
- Varying the one expert rating (rye field passes) by ±1 does not change the ranking.
- **Excluded:** water use and soil health have too few sourced trait values (2 of 5 and 6 of 20), so they are
  not scored, and their slider weight moves to the slider's other criterion.
- **Diversity check** (more hay/oats/small grain → smaller loss than the model expected?): **null or unclear.**
  All years: +0.7 points per +10 points of diverse share (90% CI −0.5 to 1.9). 2012 and 2020 point in opposite
  directions, and county diverse-crop shares are tiny (median 0.2%) and sparse.

## Data

All fetched once into `data/raw/`, processed into `data/processed/*.parquet`. The app runs fully offline.
Join key: 5-digit county FIPS + year (+ month).

| ID | Source | Use | Status |
|---|---|---|---|
| D1 | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) county corn-grain and soybean yields 2000-2025 | Target | used |
| D2 | [NASA POWER](https://power.larc.nasa.gov/) daily (AG): T2M_MAX, T2M_MIN, PRECTOTCORR, GWETROOT, GWETTOP, EVPTRNS, RH2M, WS2M | Features, calendar | used |
| D3 | [Census cartographic boundaries 2023](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip) | Centroids (EPSG:5070), map | used |
| D6 | [NASS Crop Progress (Iowa, weekly)] (https://www.nass.usda.gov/Publications/National_Crop_Progress/) | Crop calendar | used |
| D7 | NASS county hay, oats, wheat, barley acres | Diversity check | used |
| D8 | NASS Census of Agriculture cover-crop acres | Context (2017, 2022 only; 2012 returned no rows) | used |
| D9 | [USDA-NRCS Soil Data Access (SSURGO)](https://sdmdataaccess.sc.egov.usda.gov/) | Story County soil panel (descriptive) | used |
| D11 | [NASA MODIS NDVI MOD13Q1 v061 via ORNL DAAC](https://modis.ornl.gov/rst/api/v1/) | Failure panel | used |
| D12 | NASA SMAP L4 | POWER soil-wetness validation | **blocked: needs Earthdata login** |
| D15 | [US Drought Monitor](https://droughtmonitor.unl.edu/) county statistics | 2012 / 2020 check | used |
| D16 | [NOAA SPC storm reports, 10 Aug 2020](https://www.spc.noaa.gov/climo/reports/200810_rpts.html) | 2020 derecho | used |
| D17 | [USDA RMA cause of loss](https://www.rma.usda.gov/tools-reports/summary-of-business/cause-loss) 2008-2025 | Wind vs drought losses | used |
| D19 | [ISU Ag Decision Maker crop budgets](https://www.extension.iastate.edu/agdm/crops/html/a1-20.html) | Costs, labour (information gaps) | used |

Reference documents for the crop calendar, traits and strategy cards are in [data/sources.csv](data/sources.csv)
(S01-S13). Each was opened and summarised in our own words. None is marked human-verified yet.
**Cut (future work):** GPM IMERG, GRACE/GRACE-FO, SRTM.

## Method

1. **Features (B8):** NASA POWER daily → one vector per county-year from in-season windows only: July mean max
   temperature, Jul-Aug days above 30 °C, Jun-Aug rain, August and July root-zone wetness, Jun-Aug water balance
   (rain - EVPTRNS/2.45), August night temperature, late-spring frost days, May-Jun heavy-rain days (>25 mm),
   growing degree days. Sets: S0 (3), S1 (5), S2 (10). No county ID, year or coordinates.
2. **Target:** (yield - county trend) / trend × 100. The trend is a per-county OLS line fit **inside every CV fold
   on training years only**. For test year Y it is fit on years before Y (expanding); a fixed 2000-2019 trend is the
   sensitivity check.
3. **Split:** development years 2000-2019 (all exploration, tuning and selection), leave-one-year-out CV (all
   counties of a year held out together). Test year 2020-2025.
4. **Model ladder (B9):** M0 zero anomaly, M1 persistence, M2 3-feature ridge, M3 ridge S1/S2, M4 elastic net,
   M5 ridge with heat-hinge and heat × dryness, M6 monotone GAM, M7 constrained LightGBM (9 random configs),
   M8 shallow random forest; E1/E2/C1 only if their pre-declared triggers held. 18 tuned configs per crop, all logged.
5. **Selection rule** ([reports/selection_rule.md](reports/selection_rule.md)), written before the first CV result.
6. **Checks:** train-vs-CV gap, placebo (weather shuffled across years), spatial block hold-out (5 k-means regions),
   expanding-window CV 2008-2019, ablations A1 (no GWETROOT), A2 (pooled crops), A3 (Theil-Sen trend), A4
   (climatology-anomaly features).
7. **Uncertainty:** 2,000-draw year bootstrap on every metric; 90% split-conformal prediction intervals from LOYO
   residuals with coverage checked on held-out years; quantile LightGBM as a comparison.
8. **Final fit:** after the single test run, the frozen spec is refit on 2000-2025 for the app (SHAP, scenario
   sliders, intervals).
9. **Rotation engine (B11):** SHAP → month weights (stress profile) → overlap with each grain crop's critical
   period from NASS Crop Progress; bare-soil share and heavy-rain exposure of bare soil; rotation-level income and
   labour evidence. Criteria with fewer than half their trait values sourced (water use, soil health) are
   excluded and shown as "insufficient sourced data". Weight sensitivity: 2,000 Dirichlet priority mixes. Calendar
   sensitivity: every window shifted ±14 and ±21 days. Trait sensitivity: the one expert rating ±1.

## What we added (our own idea)

- **Model-to-calendar link:** the ML model's SHAP values become a month-by-month stress profile, and rotations are
  compared by how much of their crops' critical periods fall in those months. This is labelled as *exposure*, not
  predicted yield.
- **Weight-sensitivity view:** the share of 2,000 random priority mixes in which each strategy ranks first, so the
  ranking never looks arbitrary.
- **Honest gaps:** trait cells without an opened source stay `NEEDS_SOURCE`, and criteria built mostly on them are
  dropped from the score instead of being filled with defaults.

## Limitations

- Hindcast, not a forecast; features are season summaries known only after harvest.
- County scale (one POWER point per county; only ~62 distinct grid cells for 99 counties), not field scale.
- About 20 independent weather years for development and 6 test years: evidence, not proof.
- POWER is partly modelled; GWETROOT is not a direct satellite measurement. SMAP validation not done (no login).
- Scenario sliders shift temperature and rain only. Soil wetness and ET are not re-simulated, and values are
  clipped to the training range.
- RMA data are insurance claims, not measured yield loss.
- Oats, red clover and cereal rye are not modelled. Their timing uses the corn/soybean average stress profile as
  a proxy.
- Rotation scores are a transparent heuristic from cited sources and calendar exposure, not a predicted yield.
- **Test-period exposure:** an earlier prototype scored 2020-2024 before this protocol existed
  ([reports/provenance.md](reports/provenance.md)). 2025 is reported separately as the only fully unseen year.

## AI-tool disclosure

| Tool | Used for | Where |
|---|---|---|
| Claude Code (Anthropic, Claude Opus 5.5) | Helped write/debug and run the code; located and summarised sources; drafted this README | All of `src/`, `app.py`, `web/`, `tests/`, `notebooks/`, `reports/`, `data/*.csv` |
| Gemini (Google), optional | Rewrites a computed strategy card in plain language; rejected if it adds any number | Streamlit fallback only, when `GEMINI_API_KEY` is set. Not used by the website (a browser-side key would be exposed) |

Human checks are tracked in [reports/provenance.md](reports/provenance.md).

Testing Site at: https://rotate-wise.vercel.app/ (No API Key Needed)

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Put `NASS_API_KEY=...` (and optionally `GEMINI_API_KEY`, `EARTHDATA_*`) in `.env`, then run each stage
(`make <target>`, or on Windows `python -m src.run <target>`):

```bash
.venv/Scripts/python.exe -m src.run data
.venv/Scripts/python.exe -m src.run features
.venv/Scripts/python.exe -m src.run eda
.venv/Scripts/python.exe -m src.run select
.venv/Scripts/python.exe -m src.run final
.venv/Scripts/python.exe -m src.run test
.venv/Scripts/python.exe -m src.run web-data
```

`final` writes `models/final_test.lock` and refuses to run again unless `FORCE=1`, which is logged.
`web-data` writes `web/public/data/` (about 2 MB of JSON: 99 county files with a 7 × 11 what-if grid each). It takes
a few minutes and the output is committed, so the website build never needs Python.
The Colab copy is [notebooks/training.ipynb](notebooks/training.ipynb).

### Website (`web/`)

Needs Node 20 or newer.

```bash
cd web && npm ci && npm test && npm run build && npm run preview
```

`npm test` checks that the browser's scoring matches Python's exactly (`web/public/data/fixtures/scoring.json`).
To deploy on Vercel, import the repository and set **Root Directory = `web`**; `web/vercel.json` sets the build
command, the output folder and single-page routing. Fonts are self-hosted and the map is drawn from bundled county
shapes, so the site makes no requests to other hosts.

Lighthouse (mobile, 2026-11-14): accessibility 100 and best practices 100 on all 8 pages. Performance was 95-97 on
the guided pages in clean runs. The Evidence pages, which draw the charts, scored lower (roughly 75-85 in clean runs).
Timing scores varied with other load on the test machine.

Streamlit analyst view (fallback): `.venv/Scripts/python.exe -m src.run app`.

## Link audit

Every link was requested by `src/link_audit.py` (results in `reports/link_audit.csv`).

| ID | URL | Status | Checked |
|---|---|---|---|
| S01 | https://quickstats.nass.usda.gov/ | OK | 2026-11-14 |
| S02 | https://crops.extension.iastate.edu/encyclopedia/crop-water-use-or-evapotranspiration | OK | 2026-11-14 |
| S03 | https://crops.extension.iastate.edu/cropnews/2017/06/corn-water-use-and-evapotranspiration | OK | 2026-11-14 |
| S04 | https://crops.extension.iastate.edu/post/corn-after-cereal-rye-cover-crop-2021 | OK | 2026-11-14 |
| S05 | https://iowaagriculture.gov/news/cereal-rye-cover-crop-seeding-date-extended-dec-1-statewide | OK | 2026-11-14 |
| S06 | https://www.sare.org/wp-content/uploads/chart2-performance-and-roles.pdf | OK | 2026-11-14 |
| S07 | https://crops.extension.iastate.edu/cropnews/2025/03/should-i-frost-seed | OK | 2026-11-14 |
| S08 | https://practicalfarmers.org/resources/research-reports/establishing-legume-forages-with-oats | OK | 2026-11-14 |
| S09 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0047149 | OK | 2026-11-14 |
| S10 | https://www.extension.iastate.edu/agdm/crops/html/a1-20.html | HTTP 403 to scripts; opened in a browser-based reader 2026-11-14 | 2026-11-14 |
| S11 | https://crops.extension.iastate.edu/encyclopedia/soybean-cyst-nematode | OK | 2026-11-14 |
| S12 | https://agpolicyreview.card.iastate.edu/spring-2018/do-cover-crops-pay-net-returns-cornsoybean-farmers-iowa | OK | 2026-11-14 |
| S13 | https://www.card.iastate.edu/publications/22-wp-641 | OK | 2026-11-14 |
| D1 | https://quickstats.nass.usda.gov/ | OK | 2026-11-14 |
| D2 | https://power.larc.nasa.gov/ | OK | 2026-11-14 |
| D3 | https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip | OK | 2026-11-14 |
| D7 | https://quickstats.nass.usda.gov/ | OK | 2026-11-14 |
| D8 | https://quickstats.nass.usda.gov/ | OK | 2026-11-14 |
| D9 | https://sdmdataaccess.sc.egov.usda.gov/ | OK | 2026-11-14 |
| D11 | https://modis.ornl.gov/rst/api/v1/ | timed out in the audit (slow service, frequent 502s); NDVI data downloaded through it | 2026-11-14 |
| D15 | https://droughtmonitor.unl.edu/ | OK | 2026-11-14 |
| D16 | https://www.spc.noaa.gov/climo/reports/200810_rpts.html | OK | 2026-11-14 |
| D17 | https://www.rma.usda.gov/tools-reports/summary-of-business/cause-loss | OK | 2026-11-14 |
| D16 csv | https://www.spc.noaa.gov/climo/reports/200810_rpts_wind.csv | OK (data downloaded) | 2026-11-14 |
| D17 files | https://pubfs-rma.fpac.usda.gov/pub/Web_Data_Files/Summary_of_Business/cause_of_loss/ | OK (data downloaded) | 2026-11-14 |
| D15 API | https://usdmdataservices.unl.edu/api/CountyStatistics/ | OK (data downloaded) | 2026-11-14 |

## Credits

Full data and reference links: see the Data table and Link audit above

Data: NASA POWER (NASA Langley), NASA MODIS via ORNL DAAC, USDA NASS, USDA-NRCS SSURGO, USDA RMA, US Drought
Monitor (NDMC/USDA/NOAA), NOAA SPC, US Census Bureau. References: Iowa State University Extension and Outreach,
ISU CARD, SARE, Practical Farmers of Iowa, Davis et al. (2012) PLoS ONE. Libraries: pandas, NumPy, SciPy,
scikit-learn, LightGBM, pyGAM, SHAP, GeoPandas, Streamlit, Plotly. AI tools: see above.

Python libraries

pandas: https://pandas.pydata.org/
NumPy: https://numpy.org/
SciPy: https://scipy.org/
scikit-learn: https://scikit-learn.org/
LightGBM: https://lightgbm.readthedocs.io/
pyGAM: https://pygam.readthedocs.io/
SHAP: https://shap.readthedocs.io/
GeoPandas: https://geopandas.org/
Streamlit: https://streamlit.io/
Plotly: https://plotly.com/python/

Website stack

React: https://react.dev/
TypeScript: https://www.typescriptlang.org/
Vercel: https://vercel.com/

AI tools

Claude Code: https://docs.claude.com/en/docs/claude-code/overview
Gemini API (optional, Streamlit only): https://ai.google.dev/
