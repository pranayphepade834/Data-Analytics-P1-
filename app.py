"""
Factory Reallocation & Shipping Optimization Recommendation System
Nassau Candy Distributor -- Streamlit Dashboard (Phase 6)

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

This app is fully self-contained: it retrains its own models from
nassau_candy_cleaned.csv on first load (cached), so it doesn't depend on
any pickled/joblib model files that could break across scikit-learn versions.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Factory Reallocation & Shipping Optimization",
    page_icon="\U0001F36C",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "nassau_candy_cleaned.csv"
RNG_SEED = 42

# ---------------------------------------------------------------------------
# REFERENCE DATA
# ---------------------------------------------------------------------------
FACTORY_COORDS = {
    "Lot's O' Nuts":      (32.881893, -111.768036),
    "Wicked Choccy's":    (32.076176, -81.088371),
    "Sugar Shack":        (48.119140, -96.181150),
    "Secret Factory":     (41.446333, -90.565487),
    "The Other Factory":  (35.117500, -89.971107),
}

STATE_CENTROIDS = {
    'Alabama': (32.806671, -86.791130), 'Arizona': (34.048927, -111.093735),
    'Arkansas': (34.969704, -92.373123), 'California': (36.116203, -119.681564),
    'Colorado': (39.059811, -105.311104), 'Connecticut': (41.597782, -72.755371),
    'Delaware': (39.318523, -75.507141), 'District of Columbia': (38.897438, -77.026817),
    'Florida': (27.766279, -81.686783), 'Georgia': (33.040619, -83.643074),
    'Idaho': (44.240459, -114.478828), 'Illinois': (40.349457, -88.986137),
    'Indiana': (39.849426, -86.258278), 'Iowa': (42.011539, -93.210526),
    'Kansas': (38.526600, -96.726486), 'Kentucky': (37.668140, -84.670067),
    'Louisiana': (31.169546, -91.867805), 'Maine': (44.693947, -69.381927),
    'Maryland': (39.063946, -76.802101), 'Massachusetts': (42.230171, -71.530106),
    'Michigan': (43.326618, -84.536095), 'Minnesota': (45.694454, -93.900192),
    'Mississippi': (32.741646, -89.678696), 'Missouri': (38.456085, -92.288368),
    'Montana': (46.921925, -110.454353), 'Nebraska': (41.125370, -98.268082),
    'Nevada': (38.313515, -117.055374), 'New Hampshire': (43.452492, -71.563896),
    'New Jersey': (40.298904, -74.521011), 'New Mexico': (34.840515, -106.248482),
    'New York': (42.165726, -74.948051), 'North Carolina': (35.630066, -79.806419),
    'North Dakota': (47.528912, -99.784012), 'Ohio': (40.388783, -82.764915),
    'Oklahoma': (35.565342, -96.928917), 'Oregon': (44.572021, -122.070938),
    'Pennsylvania': (40.590752, -77.209755), 'Rhode Island': (41.680893, -71.511780),
    'South Carolina': (33.856892, -80.945007), 'South Dakota': (44.299782, -99.438828),
    'Tennessee': (35.747845, -86.692345), 'Texas': (31.054487, -97.563461),
    'Utah': (40.150032, -111.862434), 'Vermont': (44.045876, -72.710686),
    'Virginia': (37.769337, -78.169968), 'Washington': (47.400902, -121.490494),
    'West Virginia': (38.491226, -80.954453), 'Wisconsin': (44.268543, -89.616508),
    'Wyoming': (42.755966, -107.302490),
    'Ontario': (43.700000, -79.400000), 'Quebec': (45.500000, -73.600000),
    'British Columbia': (49.280000, -123.120000), 'Alberta': (51.050000, -114.070000),
    'Manitoba': (49.900000, -97.140000), 'Saskatchewan': (50.450000, -104.600000),
    'Nova Scotia': (44.650000, -63.570000), 'New Brunswick': (45.960000, -66.640000),
    'Newfoundland and Labrador': (47.560000, -52.710000),
    'Prince Edward Island': (46.240000, -63.130000),
}

SHIP_COST_PER_KM_PER_UNIT = 0.004  # documented assumption -- see README

FACTORY_COLORS = {
    "Lot's O' Nuts": "#e76f51", "Wicked Choccy's": "#f4a261",
    "Sugar Shack": "#2a9d8f", "Secret Factory": "#264653",
    "The Other Factory": "#8ab17d",
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Dest Lat'] = df['State/Province'].map(lambda s: STATE_CENTROIDS[s][0])
    df['Dest Lon'] = df['State/Province'].map(lambda s: STATE_CENTROIDS[s][1])
    return df


# ---------------------------------------------------------------------------
# MODEL TRAINING (cached -- runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models (one-time setup)...")
def train_models(df):
    model_df = df.copy()
    cat_cols = ['Ship Mode', 'Region', 'Division', 'Current Factory']
    model_df = pd.get_dummies(model_df, columns=cat_cols, prefix=cat_cols)

    num_cols = ['Sales', 'Units', 'Gross Profit', 'Cost', 'Distance (km)']
    scaler = StandardScaler()
    model_df[num_cols] = scaler.fit_transform(model_df[num_cols])

    drop_cols = ['Row ID', 'Order ID', 'Order Date', 'Customer ID', 'City',
                 'State/Province', 'Postal Code', 'Product ID', 'Product Name',
                 'Country/Region', 'Dest Lat', 'Dest Lon']
    drop_cols = [c for c in drop_cols if c in model_df.columns]
    model_df = model_df.drop(columns=drop_cols)

    TARGET = 'Lead Time (days)'
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=RNG_SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=RNG_SEED),
    }

    metrics = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        metrics[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'MAE': mean_absolute_error(y_test, preds),
            'R2': r2_score(y_test, preds),
        }
        # Refit on full data for deployment-quality predictions
        m.fit(X, y)

    return models, scaler, feature_names, num_cols, metrics


def build_feature_row(row, alt_factory, alt_distance, feature_names, scaler, num_cols):
    data = {c: 0 for c in feature_names}
    num_vals = scaler.transform(
        [[row['Sales'], row['Units'], row['Gross Profit'], row['Cost'], alt_distance]]
    )[0]
    for col, val in zip(num_cols, num_vals):
        data[col] = val
    for prefix, val in [('Ship Mode', row['Ship Mode']), ('Region', row['Region']),
                         ('Division', row['Division']), ('Current Factory', alt_factory)]:
        col = f"{prefix}_{val}"
        if col in data:
            data[col] = 1
    return [data[c] for c in feature_names]


def predict_for_factory(order_subset, factory, model, scaler, feature_names, num_cols):
    lat, lon = FACTORY_COORDS[factory]
    distances = order_subset.apply(
        lambda r: haversine_km(r['Dest Lat'], r['Dest Lon'], lat, lon), axis=1
    ).values
    X = np.array([
        build_feature_row(r, factory, d, feature_names, scaler, num_cols)
        for (_, r), d in zip(order_subset.iterrows(), distances)
    ])
    preds = model.predict(X)
    return preds, distances


# ---------------------------------------------------------------------------
# SCENARIO COMPUTATION (cached per filter combination -- NOT per slider move)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Simulating factory scenarios for the current filters...")
def compute_portfolio_scenarios(_df, regions, ship_modes, _models, _scaler, _feature_names, _num_cols):
    filtered = _df[_df['Region'].isin(regions) & _df['Ship Mode'].isin(ship_modes)]
    gbm = _models['Gradient Boosting']
    rows = []

    for product in sorted(filtered['Product Name'].unique()):
        prod_df = filtered[filtered['Product Name'] == product]
        if prod_df.empty:
            continue
        current_factory = prod_df['Current Factory'].iloc[0]
        division = prod_df['Division'].iloc[0]
        n_orders = len(prod_df)
        total_units = prod_df['Units'].sum()
        total_gp = prod_df['Gross Profit'].sum()

        current_preds_by_model = {
            name: predict_for_factory(prod_df, current_factory, m, _scaler, _feature_names, _num_cols)[0]
            for name, m in _models.items()
        }
        current_mean_lead = current_preds_by_model['Gradient Boosting'].mean()
        current_dist = predict_for_factory(prod_df, current_factory, gbm, _scaler, _feature_names, _num_cols)[1]
        current_mean_dist = current_dist.mean()

        for factory in FACTORY_COORDS:
            if factory == current_factory:
                continue

            alt_preds_by_model = {}
            alt_mean_dist = None
            for name, m in _models.items():
                preds, dist = predict_for_factory(prod_df, factory, m, _scaler, _feature_names, _num_cols)
                alt_preds_by_model[name] = preds
                if name == 'Gradient Boosting':
                    alt_mean_dist = dist.mean()

            mean_alt_lead = alt_preds_by_model['Gradient Boosting'].mean()
            lead_time_delta = mean_alt_lead - current_mean_lead
            lead_time_reduction_pct = -(lead_time_delta / current_mean_lead * 100) if current_mean_lead else 0
            dist_delta = alt_mean_dist - current_mean_dist
            profit_impact = -dist_delta * SHIP_COST_PER_KM_PER_UNIT * total_units
            risk_std = (alt_preds_by_model['Gradient Boosting'] - current_preds_by_model['Gradient Boosting']).std()

            model_mean_deltas = [
                alt_preds_by_model[name].mean() - current_preds_by_model[name].mean()
                for name in _models
            ]
            agreement_std = float(np.std(model_mean_deltas))

            rows.append({
                'Product Name': product, 'Division': division,
                'Current Factory': current_factory, 'Alternate Factory': factory,
                'Order Count': n_orders, 'Total Units': total_units, 'Total Gross Profit': total_gp,
                'Current Mean Lead Time (d)': round(current_mean_lead, 2),
                'Alt Mean Lead Time (d)': round(mean_alt_lead, 2),
                'Lead Time Reduction (%)': round(lead_time_reduction_pct, 2),
                'Current Mean Distance (km)': round(current_mean_dist, 1),
                'Alt Mean Distance (km)': round(alt_mean_dist, 1),
                'Estimated Profit Impact ($)': round(profit_impact, 2),
                'Risk (StdDev)': round(risk_std, 3),
                'Model Agreement StdDev': round(agreement_std, 3),
            })

    return pd.DataFrame(rows)


def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)


def score_scenarios(scenario_df, priority):
    """Fast, pure-pandas reweighting -- runs instantly on every slider move."""
    if scenario_df.empty:
        return scenario_df

    w_lead = 0.7 * (1 - priority / 100)
    w_profit = 0.7 * (priority / 100)
    w_stability = 0.2
    w_confidence = 0.1

    max_risk = scenario_df['Risk (StdDev)'].max()
    scenario_df['Stability Score'] = (
        100 * (1 - scenario_df['Risk (StdDev)'] / max_risk) if max_risk > 0 else 100
    )
    max_dis = scenario_df['Model Agreement StdDev'].max()
    scenario_df['Confidence Score'] = (
        100 * (1 - scenario_df['Model Agreement StdDev'] / max_dis) if max_dis > 0 else 100
    )

    scored = []
    for product, group in scenario_df.groupby('Product Name'):
        norm_lead = minmax(group['Lead Time Reduction (%)'])
        norm_profit = minmax(group['Estimated Profit Impact ($)'])
        norm_stability = minmax(group['Stability Score'])
        norm_confidence = minmax(group['Confidence Score'])
        score = (w_lead * norm_lead + w_profit * norm_profit +
                 w_stability * norm_stability + w_confidence * norm_confidence)
        for idx, val in score.items():
            scored.append((idx, val))

    scenario_df['Composite Score'] = scenario_df.index.map(pd.Series(dict(scored)))
    return scenario_df


def get_recommendations(scored_df):
    """Best alternate per product, or 'Keep Current' if nothing beats it."""
    if scored_df.empty:
        return pd.DataFrame()
    recs = []
    for product, group in scored_df.groupby('Product Name'):
        best = group.loc[group['Composite Score'].idxmax()]
        keep_current = best['Lead Time Reduction (%)'] <= 0
        recs.append({
            'Product Name': product, 'Division': best['Division'],
            'Current Factory': best['Current Factory'],
            'Recommended Factory': best['Current Factory'] if keep_current else best['Alternate Factory'],
            'Action': 'Keep Current' if keep_current else 'Reassign',
            'Order Count': best['Order Count'],
            'Lead Time Reduction (%)': 0.0 if keep_current else best['Lead Time Reduction (%)'],
            'Estimated Profit Impact ($)': 0.0 if keep_current else best['Estimated Profit Impact ($)'],
            'Stability Score': best['Stability Score'],
            'Confidence Score': best['Confidence Score'],
            'Composite Score': best['Composite Score'],
        })
    return pd.DataFrame(recs).sort_values('Composite Score', ascending=False)


# ---------------------------------------------------------------------------
# LOAD DATA + TRAIN MODELS
# ---------------------------------------------------------------------------
df = load_data()
models, scaler, feature_names, num_cols, model_metrics = train_models(df)

ALL_PRODUCTS = sorted(df['Product Name'].unique())
ALL_REGIONS = sorted(df['Region'].unique())
ALL_SHIP_MODES = sorted(df['Ship Mode'].unique())

# ---------------------------------------------------------------------------
# SIDEBAR -- USER CAPABILITIES
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F36C Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")
st.sidebar.divider()

selected_product = st.sidebar.selectbox("Product", ALL_PRODUCTS)
selected_regions = st.sidebar.multiselect("Region", ALL_REGIONS, default=ALL_REGIONS)
selected_ship_modes = st.sidebar.multiselect("Ship Mode", ALL_SHIP_MODES, default=ALL_SHIP_MODES)

st.sidebar.markdown("**Optimization Priority**")
priority = st.sidebar.slider(
    "Speed \u2b05\ufe0f \u2192 \ufe0f Profit", 0, 100, 50,
    help="0 = pure lead-time minimization, 100 = pure profit maximization. "
         "Stability and model-confidence always retain a small fixed weight."
)

if not selected_regions:
    selected_regions = ALL_REGIONS
if not selected_ship_modes:
    selected_ship_modes = ALL_SHIP_MODES

st.sidebar.divider()
with st.sidebar.expander("About this model"):
    m = model_metrics['Gradient Boosting']
    st.caption(
        f"Predictions use a Gradient Boosting Regressor "
        f"(RMSE {m['RMSE']:.2f}d, MAE {m['MAE']:.2f}d, R\u00b2 {m['R2']:.3f} on held-out data). "
        f"Lead Time in this dataset is a documented synthetic construction "
        f"(Ship Mode + distance + factory efficiency + noise) since the raw "
        f"Order/Ship Date fields were not usable -- see README for details."
    )

# ---------------------------------------------------------------------------
# COMPUTE SCENARIOS (cached) + SCORE (live)
# ---------------------------------------------------------------------------
scenario_df = compute_portfolio_scenarios(
    df, tuple(selected_regions), tuple(selected_ship_modes),
    models, scaler, feature_names, num_cols
)
scored_df = score_scenarios(scenario_df.copy(), priority)
rec_df = get_recommendations(scored_df)

# ---------------------------------------------------------------------------
# TOP-LEVEL KPI STRIP
# ---------------------------------------------------------------------------
st.title("Factory Reallocation & Shipping Optimization")
st.caption("Recommendation System for Nassau Candy Distributor")

if not rec_df.empty:
    reassign_df = rec_df[rec_df['Action'] == 'Reassign']
    coverage = 100 * len(reassign_df) / len(rec_df)
    avg_lt_reduction = reassign_df['Lead Time Reduction (%)'].mean() if len(reassign_df) else 0
    avg_confidence = reassign_df['Confidence Score'].mean() if len(reassign_df) else 0
    total_profit = reassign_df['Estimated Profit Impact ($)'].sum() if len(reassign_df) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recommendation Coverage", f"{coverage:.0f}%",
              help="% of products with a beneficial reassignment under current filters/priority")
    k2.metric("Avg. Lead Time Reduction", f"{avg_lt_reduction:.1f}%")
    k3.metric("Avg. Scenario Confidence", f"{avg_confidence:.0f}/100")
    k4.metric("Total Profit Opportunity", f"${total_profit:,.0f}")
else:
    st.info("No orders match the current Region / Ship Mode filters.")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F3ED Factory Optimization Simulator",
    "\U0001F504 What-If Scenario Analysis",
    "\U0001F4CA Recommendation Dashboard",
    "\u26A0\uFE0F Risk & Impact Panel",
])

# ===========================================================================
# TAB 1 -- FACTORY OPTIMIZATION SIMULATOR
# ===========================================================================
with tab1:
    st.subheader(f"Predicted performance across all factories -- {selected_product}")

    prod_scenarios = scored_df[scored_df['Product Name'] == selected_product]
    prod_orders = df[
        (df['Product Name'] == selected_product) &
        (df['Region'].isin(selected_regions)) &
        (df['Ship Mode'].isin(selected_ship_modes))
    ]

    if prod_orders.empty:
        st.warning("No orders for this product under the current filters.")
    else:
        current_factory = prod_orders['Current Factory'].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Factory", current_factory)
        c2.metric("Orders (filtered)", f"{len(prod_orders):,}")
        c3.metric("Division", prod_orders['Division'].iloc[0])

        # Build a full 5-factory comparison table (current + 4 alternates)
        current_row = prod_scenarios.iloc[0] if not prod_scenarios.empty else None
        factory_rows = []
        if current_row is not None:
            factory_rows.append({
                'Factory': current_factory,
                'Predicted Lead Time (d)': current_row['Current Mean Lead Time (d)'],
                'Distance (km)': current_row['Current Mean Distance (km)'],
                'Is Current': True,
            })
            for _, r in prod_scenarios.iterrows():
                factory_rows.append({
                    'Factory': r['Alternate Factory'],
                    'Predicted Lead Time (d)': r['Alt Mean Lead Time (d)'],
                    'Distance (km)': r['Alt Mean Distance (km)'],
                    'Is Current': False,
                })

        factory_comparison = pd.DataFrame(factory_rows)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig = px.bar(
                factory_comparison, x='Factory', y='Predicted Lead Time (d)',
                color='Is Current', color_discrete_map={True: '#e76f51', False: '#2a9d8f'},
                title="Predicted Lead Time by Factory",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with col_chart2:
            fig2 = px.bar(
                factory_comparison, x='Factory', y='Distance (km)',
                color='Is Current', color_discrete_map={True: '#e76f51', False: '#2a9d8f'},
                title="Shipping Distance by Factory",
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width='stretch')

        st.caption("\U0001F534 Current factory shown in orange, alternates in teal.")
        st.dataframe(
            factory_comparison.drop(columns=['Is Current']).sort_values('Predicted Lead Time (d)'),
            width='stretch', hide_index=True,
        )

# ===========================================================================
# TAB 2 -- WHAT-IF SCENARIO ANALYSIS
# ===========================================================================
with tab2:
    st.subheader(f"Current vs. Recommended Assignment -- {selected_product}")

    prod_rec = rec_df[rec_df['Product Name'] == selected_product]
    prod_scenarios = scored_df[scored_df['Product Name'] == selected_product]

    if prod_rec.empty:
        st.warning("No data for this product under the current filters.")
    else:
        rec = prod_rec.iloc[0]
        if rec['Action'] == 'Keep Current':
            st.success(
                f"**{selected_product}** is already at its best factory "
                f"(**{rec['Current Factory']}**) under the current priority setting."
            )
        else:
            best_scenario = prod_scenarios[
                prod_scenarios['Alternate Factory'] == rec['Recommended Factory']
            ].iloc[0]

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted Lead Time",
                f"{best_scenario['Alt Mean Lead Time (d)']:.2f}d",
                delta=f"{-best_scenario['Lead Time Reduction (%)']:.1f}%",
                delta_color="inverse",
            )
            m2.metric(
                "Shipping Distance",
                f"{best_scenario['Alt Mean Distance (km)']:.0f} km",
                delta=f"{best_scenario['Alt Mean Distance (km)'] - best_scenario['Current Mean Distance (km)']:+.0f} km",
                delta_color="inverse",
            )
            m3.metric(
                "Estimated Profit Impact",
                f"${best_scenario['Estimated Profit Impact ($)']:,.0f}",
                delta=f"${best_scenario['Estimated Profit Impact ($)']:,.0f}",
            )

            st.markdown(
                f"**Recommendation:** move **{selected_product}** from "
                f"**{rec['Current Factory']}** \u2192 **{rec['Recommended Factory']}**"
            )

            compare_df = pd.DataFrame({
                'Scenario': ['Current', 'Recommended'],
                'Lead Time (d)': [best_scenario['Current Mean Lead Time (d)'],
                                   best_scenario['Alt Mean Lead Time (d)']],
                'Distance (km)': [best_scenario['Current Mean Distance (km)'],
                                   best_scenario['Alt Mean Distance (km)']],
            })
            fig = px.bar(
                compare_df, x='Scenario', y='Lead Time (d)', color='Scenario',
                color_discrete_map={'Current': '#e76f51', 'Recommended': '#2a9d8f'},
                title="Lead Time: Current vs. Recommended", text_auto='.2f',
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

            st.caption(
                f"Stability: {best_scenario['Risk (StdDev)']:.2f}d std-dev of per-order impact | "
                f"Model Confidence: {rec['Confidence Score']:.0f}/100"
            )

        with st.expander("All alternate factories considered"):
            st.dataframe(
                prod_scenarios[[
                    'Alternate Factory', 'Alt Mean Lead Time (d)', 'Lead Time Reduction (%)',
                    'Estimated Profit Impact ($)', 'Composite Score'
                ]].sort_values('Composite Score', ascending=False),
                width='stretch', hide_index=True,
            )

# ===========================================================================
# TAB 3 -- RECOMMENDATION DASHBOARD
# ===========================================================================
with tab3:
    st.subheader("Ranked Reassignment Suggestions (all products)")
    st.caption(
        f"Ranked by composite score at current priority setting "
        f"({'Speed-focused' if priority < 40 else 'Profit-focused' if priority > 60 else 'Balanced'})."
    )

    if rec_df.empty:
        st.warning("No data under the current filters.")
    else:
        display_df = rec_df.copy()
        display_df['Composite Score'] = display_df['Composite Score'].round(3)

        st.dataframe(
            display_df[[
                'Product Name', 'Division', 'Current Factory', 'Recommended Factory', 'Action',
                'Order Count', 'Lead Time Reduction (%)', 'Estimated Profit Impact ($)',
                'Stability Score', 'Confidence Score', 'Composite Score'
            ]],
            width='stretch', hide_index=True,
        )

        reassign_df = rec_df[rec_df['Action'] == 'Reassign']
        if not reassign_df.empty:
            fig = px.bar(
                reassign_df.sort_values('Estimated Profit Impact ($)', ascending=True),
                x='Estimated Profit Impact ($)', y='Product Name', orientation='h',
                color='Estimated Profit Impact ($)', color_continuous_scale='RdYlGn',
                title="Expected Profit Impact by Recommended Reassignment",
            )
            st.plotly_chart(fig, width='stretch')

        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("\U0001F4E5 Download recommendations as CSV", csv,
                            "nassau_candy_recommendations.csv", "text/csv")

# ===========================================================================
# TAB 4 -- RISK & IMPACT PANEL
# ===========================================================================
with tab4:
    st.subheader("Profit Impact Alerts & High-Risk Warnings")

    if scored_df.empty:
        st.warning("No data under the current filters.")
    else:
        reassign_df = rec_df[rec_df['Action'] == 'Reassign']
        negative_profit = reassign_df[reassign_df['Estimated Profit Impact ($)'] < 0]
        low_stability = reassign_df[reassign_df['Stability Score'] < 50]
        low_confidence = reassign_df[reassign_df['Confidence Score'] < 50]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### \U0001F4B0 Profit Impact Alerts")
            if negative_profit.empty:
                st.success("No recommended reassignments show a negative profit impact.")
            else:
                for _, r in negative_profit.iterrows():
                    st.error(
                        f"**{r['Product Name']}** \u2192 {r['Recommended Factory']}: "
                        f"estimated **${r['Estimated Profit Impact ($)']:,.0f}** profit impact "
                        f"despite {r['Lead Time Reduction (%)']:.1f}% faster lead time. "
                        f"Consider raising the Profit priority slider before adopting."
                    )
        with col2:
            st.markdown("##### \u26A0\uFE0F High-Risk Reassignment Warnings")
            if low_stability.empty and low_confidence.empty:
                st.success("No recommended reassignments show elevated risk or low model confidence.")
            else:
                flagged = pd.concat([low_stability, low_confidence]).drop_duplicates(subset=['Product Name'])
                for _, r in flagged.iterrows():
                    st.warning(
                        f"**{r['Product Name']}** \u2192 {r['Recommended Factory']}: "
                        f"stability {r['Stability Score']:.0f}/100, "
                        f"confidence {r['Confidence Score']:.0f}/100. "
                        f"Impact is more variable across orders / less agreed-upon across models."
                    )

        st.divider()
        st.markdown("##### Stability vs. Confidence (all recommended reassignments)")
        if not reassign_df.empty:
            fig = px.scatter(
                reassign_df, x='Stability Score', y='Confidence Score',
                size=reassign_df['Estimated Profit Impact ($)'].abs().clip(lower=1),
                color='Estimated Profit Impact ($)', color_continuous_scale='RdYlGn',
                hover_name='Product Name',
                title="Bubble size = |profit impact|, color = profit impact ($)",
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray")
            fig.add_vline(x=50, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Bottom-left quadrant = higher risk (low stability, low model agreement) -- "
                "review these manually before implementing."
            )

st.divider()
st.caption(
    "Nassau Candy Factory Reallocation & Shipping Optimization Recommendation System \u2014 "
    "Phase 6 dashboard. Lead Time is a documented synthetic construction; shipping-cost "
    "sensitivity ($0.004/unit/km) is an illustrative assumption -- see README.md for full methodology."
)
