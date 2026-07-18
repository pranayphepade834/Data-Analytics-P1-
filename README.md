# Nassau Candy — Factory Reallocation & Shipping Optimization Dashboard

Phase 6 deliverable: an interactive Streamlit app that simulates factory
reassignment scenarios live, ranks them, and flags risk — built on top of
the Phase 1–5 analysis (data prep, predictive modeling, clustering,
simulation, optimization).

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Keep `nassau_candy_cleaned.csv` in the **same folder** as `app.py` — the
app reads it by relative path.

The app is fully self-contained: on first load it retrains its own Linear
Regression / Random Forest / Gradient Boosting models directly from the
CSV (cached for the rest of the session), so it doesn't depend on any
pickled model files that could break across scikit-learn versions on your
machine.

## What's in the dashboard

**Sidebar (user controls):**
- **Product** — selects the product shown in the Simulator and What-If tabs
- **Region** / **Ship Mode** — filter which historical orders feed every
  computation across all four tabs
- **Optimization Priority slider** — 0 = pure lead-time minimization,
  100 = pure profit maximization. Moving it re-ranks instantly (no
  recomputation of model predictions — only the composite-score weights
  change), while filter changes trigger a short "Simulating..." recompute.

**Tabs:**
1. **Factory Optimization Simulator** — predicted lead time & distance for
   the selected product across all 5 factories (current highlighted).
2. **What-If Scenario Analysis** — current vs. best-recommended factory,
   with lead time / distance / profit deltas.
3. **Recommendation Dashboard** — ranked reassignment suggestions for all
   15 products, downloadable as CSV.
4. **Risk & Impact Panel** — flags any recommended reassignment with a
   negative profit impact, and any with low stability or low agreement
   across the 3 underlying models.

## Methodology notes & limitations (please read before presenting results)

**Lead Time is a documented synthetic construction, not the raw data.**
The source dataset's `Order Date` / `Ship Date` fields are not usable as a
real lead-time signal (mean gap ~1,321 days, no variation by Ship Mode —
they appear to have been generated independently). Lead Time here is
instead built from: Ship Mode base speed + distance sensitivity + a small
per-factory efficiency effect + mild seasonal congestion + random noise.
Ship Mode and Distance are legitimate, learnable drivers, so the model and
rankings are methodologically sound — but treat absolute day counts as
illustrative rather than literal historical fact.

**Profit impact uses a distance-based shipping-cost proxy** ($0.004 of
profit erosion per km per unit shipped), since the dataset has no actual
per-order freight cost field. The *ranking* of factory options is
reliable (driven by real distance geometry), but the *dollar magnitudes*
should be recalibrated against real freight-cost data before being used
in a financial decision.

**Destination geography is at the State/Province level** (centroid
coordinates), not exact city/ZIP — a reasonable simplification given the
project's own framing of the modeling problem as "destination region."

**Scenario Confidence Score** reflects agreement across three
independently-trained models (Linear Regression, Random Forest, Gradient
Boosting) on the same scenario — low agreement is a genuine signal that a
recommendation is less certain, not just decoration.

## Files

- `app.py` — the dashboard
- `nassau_candy_cleaned.csv` — cleaned, feature-engineered order-level data
  (output of Phase 1)
- `requirements.txt` — Python dependencies
