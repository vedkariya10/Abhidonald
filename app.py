import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeroPlastic India — Intelligence Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AGE_ORDER       = ["18-24","25-34","35-44","45-54","55+"]
INCOME_ORDER    = ["lt25k","25-50k","50-100k","100-200k","gt200k"]
INCOME_LABELS   = {"lt25k":"<₹25K","25-50k":"₹25–50K","50-100k":"₹50–100K","100-200k":"₹1–2L","gt200k":">₹2L"}
HABIT_ORDER     = ["lt6m","6m-2y","2-5y","gt5y","No_fixed"]
PREMIUM_ORDER   = ["0","1-10","11-20","21-30","gt30"]
SUB_ORDER       = ["No","Try_first","Yes_if_discount","Yes_already"]
CITY_ORDER      = ["Tier3","Tier2","Tier1","Metro"]
PRIOR_ORDER     = ["No_not_priority","No_considered","Yes_few","Yes_regularly"]
GREEN_ID_ORDER  = ["Not_priority","Curious","Occasional","Conscious_buyer","Core_identity"]
SCENARIO_ORDER  = ["No","Unlikely","Research_first","Wishlist","Order_now"]

PERSONA_COLORS = {
    "Eco Warrior":      "#1D9E75",
    "Mindful Parent":   "#185FA5",
    "Eco Skeptic":      "#534AB7",
    "Urban Renter":     "#BA7517",
    "Price Pragmatist": "#888780",
}
CLUSTER_PAL = ["#1D9E75","#185FA5","#534AB7","#BA7517","#888780","#A32D2D"]

NUMERIC_COLS = [
    "eco_concern_score","greenwashing_skepticism","brand_switch_readiness",
    "total_hh_spend_monthly","eco_spend_willingness","lifestyle_score",
    "format_interest_count","barrier_count","trust_signal_count","aspiration_gap",
]
ORDINAL_SPECS = [
    ("age_group",            AGE_ORDER),
    ("city_tier",            CITY_ORDER),
    ("monthly_hh_income",    INCOME_ORDER),
    ("habit_strength",       HABIT_ORDER),
    ("premium_willingness_pct", PREMIUM_ORDER),
    ("subscription_intent",  SUB_ORDER),
    ("prior_eco_purchase",   PRIOR_ORDER),
    ("green_identity",       GREEN_ID_ORDER),
    ("scenario_purchase_intent", SCENARIO_ORDER),
]
NOMINAL_COLS = [
    "gender","hh_type","primary_channel","peer_eco_usage",
    "discovery_channel","preferred_eco_channel","ayurveda_affinity",
    "pricing_model_pref",
]
BINARY_COLS = [
    "format_interest_shampoo_bar","format_interest_dishwash_block",
    "format_interest_laundry_sheets","format_interest_body_bar",
    "lifestyle_exercise_yoga","lifestyle_track_waste","lifestyle_eco_content",
    "barrier_efficacy_doubt","barrier_unfamiliar","barrier_price","barrier_no_concern",
    "trust_signal_free_sample","trust_signal_certification","trust_signal_friend_rec",
    "hard_water_waxy_hair","hard_water_no_issue","sdb_flag","is_outlier",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer(df):
    df = df.copy()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Lifestyle score
    life_cols = [c for c in df.columns if c.startswith("lifestyle_") and c not in ["lifestyle_score","lifestyle_none"]]
    if life_cols:
        df["lifestyle_score"] = df[life_cols].fillna(0).apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Format interest count
    fmt_cols = [c for c in df.columns if c.startswith("format_interest_") and not c.endswith("none")]
    if fmt_cols:
        df["format_interest_count"] = df[fmt_cols].fillna(0).apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Barrier count
    barr_cols = [c for c in df.columns if c.startswith("barrier_") and not c.endswith("no_concern")]
    if barr_cols:
        df["barrier_count"] = df[barr_cols].fillna(0).apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Trust signal count
    trust_cols = [c for c in df.columns if c.startswith("trust_signal_")]
    if trust_cols:
        df["trust_signal_count"] = df[trust_cols].fillna(0).apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Aspiration gap
    if "aspiration_gap" not in df.columns:
        asp_mid = {"lt200":150,"200-400":300,"401-700":550,"701-1200":950,"gt1200":1400}
        if "actual_spend_personal_care" in df.columns:
            df["actual_spend_mid"] = df["actual_spend_personal_care"].map(asp_mid).fillna(350)
        else:
            df["actual_spend_mid"] = 350
        eco = df["eco_spend_willingness"] if "eco_spend_willingness" in df.columns else 300
        df["aspiration_gap"] = eco - df["actual_spend_mid"]

    # CPS composite
    if "overall_interest_score" in df.columns:
        df["overall_interest_score"] = pd.to_numeric(df["overall_interest_score"], errors="coerce")
        df["interest_binary"] = (df["overall_interest_score"] >= 4).astype(int)
        df["interest_3class"] = df["overall_interest_score"].apply(
            lambda x: "Not_Interested" if x <= 2 else ("Neutral" if x == 3 else "Interested")
        )

    # Eco-warrior flag
    if "green_identity" in df.columns:
        df["is_eco_warrior"] = (df["green_identity"] == "Core_identity").astype(int)

    # Subscription flag
    if "subscription_intent" in df.columns:
        df["will_subscribe"] = df["subscription_intent"].isin(["Yes_already","Yes_if_discount"]).astype(int)

    return df


@st.cache_data
def load_data():
    df = pd.read_csv("zeroplastic_dataset.csv")
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == 0].copy()
    return engineer(df)


def build_preprocessor(df, exclude_cols=None):
    exclude_cols = exclude_cols or []
    num_feats = [c for c in NUMERIC_COLS    if c in df.columns and c not in exclude_cols]
    ord_feats = [c for c,_ in ORDINAL_SPECS if c in df.columns and c not in exclude_cols]
    ord_cats  = [cats for c,cats in ORDINAL_SPECS if c in df.columns and c not in exclude_cols]
    nom_feats = [c for c in NOMINAL_COLS    if c in df.columns and c not in exclude_cols]
    bin_feats = [c for c in BINARY_COLS     if c in df.columns and c not in exclude_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    ord_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(categories=ord_cats, handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    nom_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])

    transformers = []
    if num_feats: transformers.append(("num", num_pipe, num_feats))
    if ord_feats: transformers.append(("ord", ord_pipe, ord_feats))
    if nom_feats: transformers.append(("nom", nom_pipe, nom_feats))
    if bin_feats: transformers.append(("bin", bin_pipe, bin_feats))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    all_feats = num_feats + ord_feats + nom_feats + bin_feats
    return pre, all_feats


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models():
    df = load_data()
    pre, feats = build_preprocessor(df)
    feat_cols  = [c for c in feats if c in df.columns]
    X = pre.fit_transform(df[feat_cols])
    y = df["interest_binary"].values
    y_reg = df["eco_spend_willingness"].fillna(df["eco_spend_willingness"].median()).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    lr  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    gb  = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    rf.fit(X_tr, y_tr)
    lr.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    clf_models = {"Random Forest": rf, "Logistic Regression": lr, "Gradient Boosting": gb}
    if XGB_OK:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=42, eval_metric="logloss", verbosity=0)
        xgb.fit(X_tr, y_tr)
        clf_models["XGBoost"] = xgb

    rfr   = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=1.0)
    rfr.fit(X_tr_r, y_tr_r)
    ridge.fit(X_tr_r, y_tr_r)

    clust_feats = [c for c in [
        "eco_concern_score","greenwashing_skepticism","brand_switch_readiness",
        "lifestyle_score","format_interest_count","barrier_count","trust_signal_count",
        "aspiration_gap","eco_spend_willingness","is_eco_warrior","will_subscribe",
    ] if c in df.columns]
    Xc  = df[clust_feats].fillna(df[clust_feats].median())
    sc  = StandardScaler()
    Xcs = sc.fit_transform(Xc)
    km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    km5.fit(Xcs)

    return dict(
        pre=pre, feat_cols=feat_cols,
        clf_models=clf_models,
        X_te=X_te, y_te=y_te,
        rfr=rfr, ridge=ridge,
        X_te_r=X_te_r, y_te_r=y_te_r,
        km=km5, km_sc=sc, clust_feats=clust_feats,
        Xcs=Xcs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌿 ZeroPlastic India")
st.sidebar.markdown("**Market Intelligence Dashboard**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "🤖 Predictive Modelling",
    "🎯 Prescriptive Actions",
    "📥 Upload & Score New Data",
])
st.sidebar.markdown("---")
df = load_data()
st.sidebar.caption(f"{len(df):,} Indian respondents · Seed 42")
st.sidebar.markdown(f"**Interested:** {df['interest_binary'].sum():,} ({df['interest_binary'].mean()*100:.1f}%)")
st.sidebar.markdown(f"**Avg Eco-Spend:** ₹{df['eco_spend_willingness'].mean():,.0f}/mo")


# ─────────────────────────────────────────────────────────────────────────────
# KPI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def kpi(col, label, val, color):
    col.markdown(f"""<div style='background:{color}18;border-left:4px solid {color};
        border-radius:8px;padding:14px 16px;margin:2px'>
        <div style='font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase'>{label}</div>
        <div style='font-size:26px;font-weight:700;color:#111;margin-top:4px'>{val}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analysis")
    st.markdown("**Who is our addressable market?** Demographics, spending behaviour, product interest and city distribution across 2,000 Indian consumers surveyed on plastic-free home & personal care products.")

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Respondents",      f"{len(df):,}",                                    "#1D9E75")
    kpi(c2, "Positive Intent",  f"{df['interest_binary'].mean()*100:.1f}%",         "#1D9E75")
    kpi(c3, "Avg Eco-Spend",    f"₹{df['eco_spend_willingness'].mean():,.0f}",       "#BA7517")
    kpi(c4, "Avg Aspiration Gap", f"₹{df['aspiration_gap'].mean():,.0f}",           "#A32D2D")
    kpi(c5, "Avg Eco Concern",  f"{df['eco_concern_score'].mean():.2f}/5",           "#185FA5")

    st.markdown("---")
    st.markdown("### Demographics")
    c1,c2,c3 = st.columns(3)
    with c1:
        age = df["age_group"].value_counts().reindex(AGE_ORDER).reset_index()
        age.columns = ["Age","Count"]
        fig = px.bar(age, x="Age", y="Count", color="Count",
                     color_continuous_scale="Teal", title="Age distribution")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=10,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gen = df["gender"].value_counts().reset_index()
        gen.columns = ["Gender","Count"]
        fig = px.pie(gen, names="Gender", values="Count", hole=0.45,
                     title="Gender split",
                     color_discrete_sequence=["#1D9E75","#185FA5","#888780"])
        fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        city = df["city_tier"].value_counts().reset_index()
        city.columns = ["City Tier","Count"]
        fig = px.bar(city, x="Count", y="City Tier", orientation="h",
                     color="Count", color_continuous_scale="Purples", title="City tier distribution")
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                          margin=dict(t=40,b=10,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    c4,c5 = st.columns(2)
    with c4:
        inc = df["monthly_hh_income"].value_counts().reindex(INCOME_ORDER).reset_index()
        inc.columns = ["Income","Count"]
        inc["Income Label"] = inc["Income"].map(INCOME_LABELS)
        fig = px.bar(inc, x="Income Label", y="Count", color="Count",
                     color_continuous_scale="Blues", title="Monthly household income")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=10,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        if "persona" in df.columns:
            pc = df["persona"].value_counts().reset_index()
            pc.columns = ["Persona","Count"]
            fig = px.pie(pc, names="Persona", values="Count", hole=0.4,
                         title="Seeded persona distribution",
                         color="Persona", color_discrete_map=PERSONA_COLORS)
            fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            ht = df["hh_type"].value_counts().reset_index()
            ht.columns = ["Household","Count"]
            fig = px.pie(ht, names="Household", values="Count", hole=0.45,
                         title="Household type",
                         color_discrete_sequence=CLUSTER_PAL)
            fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Spending & Eco-Budget")
    c6,c7 = st.columns(2)
    with c6:
        fig = px.histogram(df, x="eco_spend_willingness", nbins=40,
                           color_discrete_sequence=["#1D9E75"],
                           title="Eco-spend willingness distribution (₹/month)",
                           labels={"eco_spend_willingness":"Eco-Spend Willingness (₹)"})
        fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c7:
        fig = px.histogram(df, x="aspiration_gap", nbins=40,
                           color_discrete_sequence=["#A32D2D"],
                           title="Aspiration gap: stated eco-spend − actual spend (₹)",
                           labels={"aspiration_gap":"Aspiration Gap (₹)"})
        fig.add_vline(x=0, line_dash="dash", line_color="#888",
                      annotation_text="Zero gap", annotation_position="top right")
        fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Product Format Interest & Barriers")
    c8,c9 = st.columns(2)
    with c8:
        fmt_cols = {
            "format_interest_shampoo_bar":    "Shampoo Bar",
            "format_interest_dishwash_block": "Dishwash Block",
            "format_interest_laundry_sheets": "Laundry Sheets",
            "format_interest_body_bar":       "Body Bar",
            "format_interest_surface_tablets":"Surface Tablets",
            "format_interest_conditioner_bar":"Conditioner Bar",
            "format_interest_toilet_tablet":  "Toilet Tablet",
        }
        present = {k:v for k,v in fmt_cols.items() if k in df.columns}
        if present:
            fm = pd.DataFrame({
                "Format": list(present.values()),
                "Interest %": [df[k].mean()*100 for k in present.keys()]
            }).sort_values("Interest %", ascending=True)
            fig = px.bar(fm, x="Interest %", y="Format", orientation="h",
                         color="Interest %", color_continuous_scale="Teal",
                         title="Plastic-free format interest rates")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=10,l=0,r=0), height=360)
            st.plotly_chart(fig, use_container_width=True)
    with c9:
        barr_cols = {
            "barrier_efficacy_doubt": "Efficacy doubt",
            "barrier_unfamiliar":     "Unfamiliar format",
            "barrier_price":          "Too expensive",
            "barrier_availability":   "Availability",
            "barrier_habit":          "Habit / loyalty",
            "barrier_humidity":       "Humidity concern",
        }
        present_b = {k:v for k,v in barr_cols.items() if k in df.columns}
        if present_b:
            br = pd.DataFrame({
                "Barrier": list(present_b.values()),
                "% Selected": [df[k].mean()*100 for k in present_b.keys()]
            }).sort_values("% Selected", ascending=True)
            fig = px.bar(br, x="% Selected", y="Barrier", orientation="h",
                         color="% Selected", color_continuous_scale="Reds",
                         title="Top purchase barriers")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=10,l=0,r=0), height=360)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### City Tier × Interest Score Heatmap")
    if "city_tier" in df.columns:
        heat = df.groupby("city_tier")["overall_interest_score"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
        heat = heat.reindex(CITY_ORDER[::-1])
        fig = px.imshow(heat, color_continuous_scale="YlGn", text_auto=".1f",
                        title="% of respondents per city tier × interest score (Q34)",
                        labels={"x":"Interest Score","y":"City Tier","color":"% of Tier"},
                        aspect="auto")
        fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Discovery Channel & Trust Signals")
    c10,c11 = st.columns(2)
    with c10:
        if "discovery_channel" in df.columns:
            dc = df["discovery_channel"].value_counts().reset_index()
            dc.columns = ["Channel","Count"]
            fig = px.bar(dc, x="Count", y="Channel", orientation="h",
                         color="Count", color_continuous_scale="Blues",
                         title="Primary discovery channel for eco-products")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                              margin=dict(t=40,b=10,l=0,r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)
    with c11:
        trust_cols = {
            "trust_signal_lab_tests":       "Lab test results",
            "trust_signal_certification":   "Eco certification",
            "trust_signal_friend_rec":      "Friend recommendation",
            "trust_signal_free_sample":     "Free sample",
            "trust_signal_money_back":      "Money-back guarantee",
            "trust_signal_reviews":         "4.5★ reviews",
        }
        present_t = {k:v for k,v in trust_cols.items() if k in df.columns}
        if present_t:
            tr = pd.DataFrame({
                "Trust Signal": list(present_t.values()),
                "% Selected": [df[k].mean()*100 for k in present_t.keys()]
            }).sort_values("% Selected", ascending=True)
            fig = px.bar(tr, x="% Selected", y="Trust Signal", orientation="h",
                         color="% Selected", color_continuous_scale="Purples",
                         title="Trust signals that would drive trial")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=10,l=0,r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Ayurveda Affinity & Subscription Intent")
    c12,c13 = st.columns(2)
    with c12:
        if "ayurveda_affinity" in df.columns:
            ay = df["ayurveda_affinity"].value_counts().reset_index()
            ay.columns = ["Affinity","Count"]
            fig = px.bar(ay, x="Count", y="Affinity", orientation="h",
                         color="Count", color_continuous_scale="Greens",
                         title="Ayurveda / natural ingredient affinity (Q25)")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                              margin=dict(t=40,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
    with c13:
        if "subscription_intent" in df.columns:
            si = df["subscription_intent"].value_counts().reset_index()
            si.columns = ["Intent","Count"]
            fig = px.pie(si, names="Intent", values="Count", hole=0.45,
                         title="Subscription / auto-refill intent (Q28)",
                         color_discrete_sequence=CLUSTER_PAL)
            fig.update_layout(margin=dict(t=40,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analysis")
    st.markdown("**Why are some eco-aware consumers not converting?** Root-cause analysis, psychographic correlations, statistical significance tests and switching triggers.")

    st.markdown("---")
    st.markdown("### Eco-Skeptic Paradox — High Awareness, Low Conversion")
    high_eco = df[df["eco_concern_score"] >= 4]
    skeptic_nc = high_eco[high_eco["overall_interest_score"] <= 2]
    eco_conv   = high_eco[high_eco["overall_interest_score"] >= 4]
    c1,c2,c3 = st.columns(3)
    kpi(c1, "High Eco Concern (Q12≥4)",    f"{len(high_eco):,}",   "#1D9E75")
    kpi(c2, "Eco-aware NOT Converted",     f"{len(skeptic_nc):,}", "#A32D2D")
    kpi(c3, "Eco-aware AND Converted",     f"{len(eco_conv):,}",   "#1D9E75")

    c4,c5 = st.columns(2)
    with c4:
        st.markdown("#### Greenwashing Skepticism: Converted vs Not")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=skeptic_nc["greenwashing_skepticism"].dropna(),
                                    name="Not Converted", marker_color="#A32D2D", opacity=0.7, nbinsx=5))
        fig.add_trace(go.Histogram(x=eco_conv["greenwashing_skepticism"].dropna(),
                                    name="Converted", marker_color="#1D9E75", opacity=0.7, nbinsx=5))
        fig.update_layout(barmode="overlay", height=300, xaxis_title="Greenwashing Skepticism (1–5)",
                           margin=dict(t=40,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        st.markdown("#### Prior Eco-Purchase: Converted vs Not")
        cats = PRIOR_ORDER
        vals_nc = [len(skeptic_nc[skeptic_nc["prior_eco_purchase"]==c]) for c in cats]
        vals_c  = [len(eco_conv[eco_conv["prior_eco_purchase"]==c]) for c in cats]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Not Converted", x=cats, y=vals_nc, marker_color="#A32D2D", opacity=0.85))
        fig.add_trace(go.Bar(name="Converted",     x=cats, y=vals_c,  marker_color="#1D9E75", opacity=0.85))
        fig.update_layout(barmode="group", height=300, xaxis_title="Prior Eco Purchase",
                           margin=dict(t=40,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Psychographic Drivers of Purchase Intent")
    ps_map = {
        "eco_concern_score":       "Eco concern score",
        "greenwashing_skepticism": "Greenwashing skepticism",
        "brand_switch_readiness":  "Brand switch readiness",
        "lifestyle_score":         "Lifestyle score",
        "trust_signal_count":      "Trust signals selected",
        "barrier_count":           "Barriers cited",
        "aspiration_gap":          "Aspiration gap (₹)",
        "format_interest_count":   "Format interest count",
    }
    corrs = []
    for c, lbl in ps_map.items():
        if c in df.columns:
            valid = df[[c,"interest_binary"]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[c], valid["interest_binary"])
                corrs.append({"Feature": lbl, "Pearson r": round(r,3), "p-value": round(p,4)})
    if corrs:
        cdf = pd.DataFrame(corrs).sort_values("Pearson r")
        colors = ["#A32D2D" if v < 0 else "#1D9E75" for v in cdf["Pearson r"]]
        fig = go.Figure(go.Bar(x=cdf["Pearson r"], y=cdf["Feature"], orientation="h",
                                marker_color=colors, text=cdf["Pearson r"], textposition="outside"))
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        fig.update_layout(title="Feature → Buy Intent correlation (Pearson r)",
                           xaxis_title="Pearson r", margin=dict(t=50,b=10,l=0,r=60), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Trust Chain Analysis — What Converts the Eco-Skeptic?")
    if "greenwashing_skepticism" in df.columns:
        df["skep_band"] = pd.cut(df["greenwashing_skepticism"],
                                  bins=[0,2,3,5], labels=["Low (1–2)","Medium (3)","High (4–5)"])
        sb = df.groupby("skep_band", observed=True)["interest_binary"].mean().reset_index()
        sb.columns = ["Skepticism Band","Buy Rate"]
        sb["Buy Rate %"] = (sb["Buy Rate"]*100).round(1)
        fig = px.bar(sb, x="Skepticism Band", y="Buy Rate %", color="Buy Rate %",
                     color_continuous_scale="RdYlGn", title="Buy rate by greenwashing skepticism band",
                     text="Buy Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=10,l=0,r=0), height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Habit Strength vs Conversion Rate")
    if "habit_strength" in df.columns:
        c6,c7 = st.columns(2)
        with c6:
            hb = df.groupby("habit_strength", observed=True)["interest_binary"].mean().reset_index()
            hb.columns = ["Habit","Buy Rate"]
            hb["Buy Rate %"] = (hb["Buy Rate"]*100).round(1)
            hb["Habit"] = pd.Categorical(hb["Habit"], categories=HABIT_ORDER, ordered=True)
            hb = hb.sort_values("Habit")
            fig = px.bar(hb, x="Habit", y="Buy Rate %", color="Buy Rate %",
                         color_continuous_scale="Teal", title="Buy rate by brand habit strength", text="Buy Rate %")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=10,l=0,r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)
        with c7:
            pb = df.groupby("peer_eco_usage", observed=True)["interest_binary"].mean().reset_index()
            pb.columns = ["Peer Usage","Buy Rate"]
            pb["Buy Rate %"] = (pb["Buy Rate"]*100).round(1)
            pb = pb.sort_values("Buy Rate %", ascending=False)
            fig = px.bar(pb, x="Buy Rate %", y="Peer Usage", orientation="h",
                         color="Buy Rate %", color_continuous_scale="Greens",
                         title="Buy rate by peer eco-product usage", text="Buy Rate %")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=10,l=0,r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Aspiration Gap Analysis")
    c8,c9 = st.columns(2)
    with c8:
        df["gap_band"] = pd.cut(df["aspiration_gap"],
                                 bins=[-2000, 0, 300, 800, 5000],
                                 labels=["Negative gap","Low (<₹300)","Medium (₹300–800)","High (>₹800)"])
        gb2 = df.groupby("gap_band", observed=True)["interest_binary"].mean().reset_index()
        gb2.columns = ["Aspiration Gap Band","Buy Rate"]
        gb2["Buy Rate %"] = (gb2["Buy Rate"]*100).round(1)
        fig = px.bar(gb2, x="Aspiration Gap Band", y="Buy Rate %", color="Buy Rate %",
                     color_continuous_scale="RdYlGn", text="Buy Rate %",
                     title="Buy rate by aspiration gap band")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=10,l=0,r=0), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c9:
        if "scenario_purchase_intent" in df.columns:
            sc2 = df.groupby("scenario_purchase_intent", observed=True)["aspiration_gap"].mean().sort_values()
            fig = go.Figure(go.Bar(x=sc2.values, y=sc2.index, orientation="h",
                                    marker_color=[("#1D9E75" if v < 300 else ("#BA7517" if v < 600 else "#A32D2D")) for v in sc2.values],
                                    text=[f"₹{v:,.0f}" for v in sc2.values], textposition="outside"))
            fig.update_layout(title="Avg aspiration gap by scenario probe answer (Q33)",
                               xaxis_title="Avg Gap (₹)", margin=dict(t=50,b=10,l=0,r=0), height=320, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Statistical Significance Tests (Chi-Square)")
    test_res = []
    target_col = "interest_binary"
    for feat in ["green_identity","prior_eco_purchase","peer_eco_usage","habit_strength",
                 "city_tier","ayurveda_affinity","subscription_intent"]:
        if feat in df.columns:
            ct = pd.crosstab(df[feat], df[target_col])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            test_res.append({
                "Feature": feat, "Chi² stat": round(chi2,2), "p-value": round(p,4),
                "DoF": dof, "Significant": ("Yes ✅" if p < 0.05 else "No ❌")
            })
    st.dataframe(pd.DataFrame(test_res), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Hard Water Impact on Shampoo Bar Interest")
    if "hard_water_waxy_hair" in df.columns and "format_interest_shampoo_bar" in df.columns:
        hw_int = df.groupby("hard_water_waxy_hair")["format_interest_shampoo_bar"].mean().reset_index()
        hw_int.columns = ["Hard Water Complaint","Shampoo Bar Interest %"]
        hw_int["Shampoo Bar Interest %"] = (hw_int["Shampoo Bar Interest %"]*100).round(1)
        hw_int["Hard Water Complaint"] = hw_int["Hard Water Complaint"].map({0:"No complaint",1:"Waxy hair complaint"})
        c10,c11 = st.columns(2)
        with c10:
            fig = px.bar(hw_int, x="Hard Water Complaint", y="Shampoo Bar Interest %",
                         color="Shampoo Bar Interest %", color_continuous_scale="RdYlGn",
                         text="Shampoo Bar Interest %",
                         title="Shampoo bar interest: hard water vs no complaint")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=50,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c11:
            st.info("**Prescription:** Hard-water complainants show 20–30% lower shampoo bar interest. Formulate a 'Hard Water Shield' variant with citric acid and label it explicitly. Target Metro/Tier1 cities (Delhi, Bengaluru, Pune) where hard-water incidence is highest.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PREDICTIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Predictive Modelling":
    st.title("🤖 Predictive Modelling")
    st.markdown("Four ML algorithms — Classification · Clustering · Association Rules · Regression")

    with st.spinner("Training models on 2,000 respondents… (~20 sec first load)"):
        m = train_all_models()

    tab1,tab2,tab3,tab4 = st.tabs(["🎯 Classification","🔵 Clustering","🔗 Association Rules","📈 Regression"])

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Classification — Will This Person Buy from ZeroPlastic India?")
        st.markdown("**Target:** `interest_binary` (Q34 ≥ 4)  ·  **Models:** Random Forest · Logistic Regression · Gradient Boosting · XGBoost")

        X_te = m["X_te"]; y_te = m["y_te"]
        results = []
        for name, mdl in m["clf_models"].items():
            yp    = mdl.predict(X_te)
            yprob = mdl.predict_proba(X_te)[:,1]
            results.append({
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_te, yp),    4),
                "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
                "Recall":    round(recall_score(y_te, yp, zero_division=0),    4),
                "F1-Score":  round(f1_score(y_te, yp, zero_division=0),        4),
                "ROC-AUC":   round(roc_auc_score(y_te, yprob),  4),
            })
        st.markdown("#### Model Comparison")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        mc     = st.selectbox("Select model for detailed charts", list(m["clf_models"].keys()), index=0)
        chosen = m["clf_models"][mc]
        yp     = chosen.predict(X_te)
        yprob  = chosen.predict_proba(X_te)[:,1]

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            cm2 = confusion_matrix(y_te, yp)
            fig = px.imshow(cm2, text_auto=True, color_continuous_scale="Teal",
                             x=["Predicted: No","Predicted: Yes"],
                             y=["Actual: No","Actual: Yes"],
                             title=f"Confusion Matrix — {mc}", aspect="auto")
            fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### ROC Curve — All Models")
            fig = go.Figure()
            for name, mdl in m["clf_models"].items():
                pp = mdl.predict_proba(X_te)[:,1]
                fpr, tpr, _ = roc_curve(y_te, pp)
                auc = roc_auc_score(y_te, pp)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                      line=dict(dash="dash", color="#aaa"), showlegend=False))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                               title="ROC Curve — All Models",
                               margin=dict(t=50,b=10,l=0,r=0), height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Feature Importance")
        feat_cols = m["feat_cols"]
        if hasattr(chosen, "feature_importances_"):
            imp = chosen.feature_importances_
        elif hasattr(chosen, "coef_"):
            imp = np.abs(chosen.coef_[0])
        else:
            imp = np.zeros(len(feat_cols))
        fi = pd.DataFrame({"Feature": feat_cols, "Importance": imp})
        fi = fi.sort_values("Importance", ascending=False).head(20)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Teal",
                     title=f"Top 20 feature importances — {mc}")
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                           margin=dict(t=50,b=10,l=0,r=0), height=540)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Classification Report")
        rep = pd.DataFrame(classification_report(y_te, yp, output_dict=True)).T.round(3)
        st.dataframe(rep, use_container_width=True)

    # ── CLUSTERING ───────────────────────────────────────────────────────────
    with tab2:
        st.subheader("K-Means Clustering — Customer Persona Discovery")
        Xcs = m["Xcs"]

        st.markdown("#### Elbow & Silhouette")
        inertias, silhouettes = [], []
        for k in range(2, 9):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl    = km_tmp.fit_predict(Xcs)
            inertias.append(km_tmp.inertia_)
            silhouettes.append(silhouette_score(Xcs, lbl))

        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=list(range(2,9)), y=inertias, mode="lines+markers",
                                        marker=dict(color="#1D9E75", size=8),
                                        line=dict(color="#1D9E75")))
            fig.add_vline(x=5, line_dash="dash", line_color="#A32D2D",
                          annotation_text="k=5 chosen")
            fig.update_layout(title="Elbow curve", xaxis_title="K", yaxis_title="Inertia",
                               margin=dict(t=50,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Scatter(x=list(range(2,9)), y=silhouettes, mode="lines+markers",
                                        marker=dict(color="#534AB7", size=8),
                                        line=dict(color="#534AB7")))
            fig.update_layout(title="Silhouette score", xaxis_title="K", yaxis_title="Score",
                               margin=dict(t=50,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

        k_val  = st.slider("Select K", 2, 8, 5, 1)
        km_sel = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = km_sel.fit_predict(Xcs)
        df_c   = load_data()
        df_c["Cluster"] = [f"Cluster {i+1}" for i in labels]

        pca   = PCA(n_components=2, random_state=42)
        Xpca  = pca.fit_transform(Xcs)
        pca_df = pd.DataFrame(Xpca, columns=["PC1","PC2"])
        pca_df["Cluster"] = df_c["Cluster"].values
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=CLUSTER_PAL,
                         title=f"PCA 2D projection (k={k_val})", opacity=0.65)
        fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Cluster Profiles")
        pc = [c for c in ["eco_concern_score","greenwashing_skepticism","lifestyle_score",
                           "format_interest_count","eco_spend_willingness","aspiration_gap",
                           "trust_signal_count","barrier_count","interest_binary"] if c in df_c.columns]
        prof = df_c.groupby("Cluster")[pc].mean().round(2)
        st.dataframe(prof, use_container_width=True)

        rf_feats = [c for c in ["eco_concern_score","greenwashing_skepticism","lifestyle_score",
                                 "eco_spend_willingness","format_interest_count","trust_signal_count"]
                    if c in prof.columns]
        fig = go.Figure()
        for i,(cl,row) in enumerate(prof.iterrows()):
            vals = [(row[f]-df_c[f].min())/(df_c[f].max()-df_c[f].min()+1e-9) for f in rf_feats]
            vals += [vals[0]]
            theta = rf_feats + [rf_feats[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill="toself", name=cl,
                                           line_color=CLUSTER_PAL[i%len(CLUSTER_PAL)], opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                          title="Cluster radar profiles",
                          margin=dict(t=60,b=20,l=20,r=20), height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── ASSOCIATION RULES ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Association Rule Mining — Product & Behaviour Baskets")
        st.markdown("Apriori algorithm ranked by **lift**, filtered by **support** and **confidence**.")

        btype = st.selectbox("Basket type", [
            "Format Interest + Lifestyle",
            "Format Interest only",
            "Trust Signals + Barriers",
            "Lifestyle + Bundle Preferences",
        ])
        c1,c2 = st.columns(2)
        with c1: msup  = st.slider("Min support",     0.01, 0.40, 0.06, 0.01)
        with c2: mconf = st.slider("Min confidence",  0.10, 0.90, 0.40, 0.05)

        def make_tx(row):
            items = []
            if btype in ["Format Interest + Lifestyle","Format Interest only"]:
                for col in [c for c in df.columns if c.startswith("format_interest_") and not c.endswith("none")]:
                    if row.get(col, 0) == 1:
                        items.append(f"Format:{col.replace('format_interest_','').replace('_',' ').title()}")
            if btype in ["Format Interest + Lifestyle","Lifestyle + Bundle Preferences"]:
                for col in [c for c in df.columns if c.startswith("lifestyle_") and c not in ["lifestyle_score","lifestyle_none"]]:
                    if row.get(col, 0) == 1:
                        items.append(f"Lifestyle:{col.replace('lifestyle_','').replace('_',' ').title()}")
            if btype == "Trust Signals + Barriers":
                for col in [c for c in df.columns if c.startswith("trust_signal_")]:
                    if row.get(col, 0) == 1:
                        items.append(f"Trust:{col.replace('trust_signal_','').replace('_',' ').title()}")
                for col in [c for c in df.columns if c.startswith("barrier_") and not c.endswith("no_concern")]:
                    if row.get(col, 0) == 1:
                        items.append(f"Barrier:{col.replace('barrier_','').replace('_',' ').title()}")
            if btype == "Lifestyle + Bundle Preferences":
                for col in [c for c in df.columns if c.startswith("bundle_pref_") and not c.endswith("no_bundle")]:
                    if row.get(col, 0) == 1:
                        items.append(f"Bundle:{col.replace('bundle_pref_','').replace('_',' ').title()}")
            return items

        txs = [t for t in df.apply(make_tx, axis=1).tolist() if len(t) >= 2]

        with st.spinner("Running Apriori…"):
            te     = TransactionEncoder()
            te_arr = te.fit_transform(txs)
            te_df  = pd.DataFrame(te_arr, columns=te.columns_)
            freq   = apriori(te_df, min_support=msup, use_colnames=True)

        if freq.empty:
            st.warning("No frequent itemsets found. Lower min support.")
        else:
            try:
                rules = association_rules(freq, metric="confidence",
                                          min_threshold=mconf, num_itemsets=len(freq))
            except TypeError:
                rules = association_rules(freq, metric="confidence", min_threshold=mconf)

            rules = rules.sort_values("lift", ascending=False)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules["consequents"]  = rules["consequents"].apply(lambda x: ", ".join(list(x)))
            for col in ["support","confidence","lift"]:
                rules[col] = rules[col].round(3)

            st.success(f"Found **{len(rules)}** rules from **{len(freq)}** frequent itemsets.")

            c1,c2 = st.columns(2)
            with c1:
                fig = px.scatter(rules.head(100), x="support", y="confidence", color="lift",
                                  size="lift", color_continuous_scale="Viridis",
                                  hover_data=["antecedents","consequents"],
                                  title="Support vs Confidence (size/colour = lift)")
                fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=360)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                top = rules.head(20).copy()
                top["Rule"] = top.apply(lambda r: f"{r['antecedents']} → {r['consequents']}", axis=1)
                fig = px.bar(top, x="lift", y="Rule", orientation="h", color="confidence",
                              color_continuous_scale="Oranges", title="Top 20 rules by lift")
                fig.update_layout(coloraxis_showscale=True, yaxis=dict(autorange="reversed"),
                                   margin=dict(t=50,b=10,l=0,r=0), height=560)
                st.plotly_chart(fig, use_container_width=True)

            # Confidence heatmap
            ta = rules["antecedents"].value_counts().head(8).index.tolist()
            tc = rules["consequents"].value_counts().head(8).index.tolist()
            heat = rules[rules["antecedents"].isin(ta) & rules["consequents"].isin(tc)]
            if not heat.empty:
                piv = heat.pivot_table(index="antecedents", columns="consequents",
                                        values="confidence", aggfunc="max").fillna(0)
                fig = px.imshow(piv, color_continuous_scale="Blues", text_auto=".2f",
                                 title="Confidence heatmap (top antecedents × consequents)", aspect="auto")
                fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### All Rules")
            disp = rules[["antecedents","consequents","support","confidence","lift"]].copy()
            disp.columns = ["Antecedents","Consequents","Support","Confidence","Lift"]
            st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=400)

    # ── REGRESSION ───────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Regression — Predicting Eco-Spend Willingness (₹/month)")
        st.markdown("**Target:** `eco_spend_willingness`  ·  **Models:** Random Forest Regressor · Ridge Regression")

        X_te_r = m["X_te_r"]; y_te_r = m["y_te_r"]
        reg_results = []
        for name, mdl in [("Random Forest Regressor", m["rfr"]), ("Ridge Regression", m["ridge"])]:
            yp = mdl.predict(X_te_r)
            reg_results.append({
                "Model": name,
                "R²":    round(r2_score(y_te_r, yp), 4),
                "MAE":   f"₹{mean_absolute_error(y_te_r, yp):,.0f}",
                "RMSE":  f"₹{np.sqrt(mean_squared_error(y_te_r, yp)):,.0f}",
            })
        st.dataframe(pd.DataFrame(reg_results), use_container_width=True, hide_index=True)

        rc       = st.selectbox("Select regression model", ["Random Forest Regressor","Ridge Regression"])
        chosen_r = m["rfr"] if rc == "Random Forest Regressor" else m["ridge"]
        yp_r     = chosen_r.predict(X_te_r)

        c1,c2 = st.columns(2)
        with c1:
            sdf = pd.DataFrame({"Actual (₹)": y_te_r, "Predicted (₹)": yp_r})
            fig = px.scatter(sdf, x="Actual (₹)", y="Predicted (₹)", opacity=0.5,
                              color_discrete_sequence=["#1D9E75"],
                              title=f"Actual vs Predicted — {rc}")
            _mf,_bf = np.polyfit(y_te_r, yp_r, 1)
            _xrf    = np.linspace(y_te_r.min(), y_te_r.max(), 50)
            fig.add_trace(go.Scatter(x=_xrf, y=_mf*_xrf+_bf, mode="lines",
                                      line=dict(color="#1D9E75", dash="dot", width=2),
                                      showlegend=False, name="Fit"))
            mn = min(y_te_r.min(), yp_r.min()); mx = max(y_te_r.max(), yp_r.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                     line=dict(dash="dash", color="#A32D2D"),
                                     showlegend=False, name="Perfect fit"))
            fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=360)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            res = y_te_r - yp_r
            fig = px.histogram(pd.DataFrame({"Residual (₹)": res}), x="Residual (₹)", nbins=40,
                                color_discrete_sequence=["#534AB7"], title="Residuals distribution")
            fig.add_vline(x=0, line_dash="dash", line_color="#A32D2D")
            fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=360)
            st.plotly_chart(fig, use_container_width=True)

        if hasattr(chosen_r, "feature_importances_"):
            fi = pd.DataFrame({"Feature": m["feat_cols"], "Importance": chosen_r.feature_importances_})
            fi = fi.sort_values("Importance", ascending=False).head(20)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Greens",
                          title=f"Top 20 predictors of eco-spend — {rc}")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                               margin=dict(t=50,b=10,l=0,r=0), height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            ci = pd.DataFrame({"Feature": m["feat_cols"], "Coefficient": np.abs(chosen_r.coef_)})
            ci = ci.sort_values("Coefficient", ascending=False).head(20)
            fig = px.bar(ci, x="Coefficient", y="Feature", orientation="h",
                          color="Coefficient", color_continuous_scale="Greens",
                          title="Ridge coefficients (absolute) — top 20")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                               margin=dict(t=50,b=10,l=0,r=0), height=520)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PRESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Prescriptive Actions":
    st.title("🎯 Prescriptive Actions")
    st.markdown("**What exact action should we take for each customer segment?** All algorithms converge into specific, actionable go-to-market strategies.")

    ACTIONS = {
        "Eco Warrior": {
            "icon": "🌿", "headline": "VIP Eco-Club — Full Home Kit + Monthly Subscription",
            "channel": "Instagram eco-influencers · D2C website · WhatsApp subscription alerts",
            "message": "100% plastic-free. Made in India. Delivered to your door.",
            "product": "Full Home Kit ₹899 · Monthly subscription at 15% off · Refill pods",
            "price": "₹600 – ₹1,500/month", "color": "#1D9E75",
            "cac": "HIGH — 80%+ conversion rate, highest LTV, best subscription retention"},
        "Mindful Parent": {
            "icon": "👨‍👩‍👧", "headline": "Safety-First Bathroom Starter Kit",
            "channel": "Nykaa listing · Parenting blogs · Amazon with A+ content",
            "message": "Safe for your family. No chemicals. No plastic. Just clean.",
            "product": "Bathroom Starter Kit ₹499 · Shampoo bar + Body bar + Conditioner bar",
            "price": "₹400 – ₹900/month", "color": "#185FA5",
            "cac": "MEDIUM-HIGH — 55–65% conversion, family repeat purchase, high basket size"},
        "Eco Skeptic": {
            "icon": "🔬", "headline": "Proof-First: Free Sample + Lab Certificate",
            "channel": "Amazon review seeding · Certification badge on website · Free trial page",
            "message": "Lab-tested. Third-party certified. No greenwashing. Ever.",
            "product": "Free Sample Kit (shampoo bar + dishwash block) · No purchase required",
            "price": "Acquire at cost — LTV ₹800+/month post-conversion", "color": "#534AB7",
            "cac": "HIGH acquisition cost, LOW churn once converted — most loyal segment"},
        "Urban Renter": {
            "icon": "🏙️", "headline": "Blinkit/Zepto Impulse Trial — ₹199 Laundry Sheets",
            "channel": "Blinkit · Zepto · Swiggy Instamart · Instagram Reels demo",
            "message": "No mess, no plastic, no waste. Delivered in 15 minutes.",
            "product": "Laundry Sheets 30-load trial ₹199 · Body bar ₹129 · Dishwash block ₹149",
            "price": "₹300 – ₹700/month", "color": "#BA7517",
            "cac": "LOW-MEDIUM — high trial rate, format virality, social sharing multiplier"},
        "Price Pragmatist": {
            "icon": "💰", "headline": "Per-Wash Cost Calculator — Value Story First",
            "channel": "Jiomart · Regional language content · Local kirana B2B",
            "message": "1 dishwash block = 3 bottles. ₹4 per wash vs ₹7. Do the maths.",
            "product": "Dishwash block single ₹149 · Value bundle 3-pack ₹399",
            "price": "₹149 – ₹400/month", "color": "#888780",
            "cac": "LOW — 12–20% conversion. Educate on per-wash cost. 12–18 month horizon."},
    }

    segs = st.multiselect("Show segments", list(ACTIONS.keys()), default=list(ACTIONS.keys()))
    for seg in segs:
        a  = ACTIONS[seg]
        if "persona" in df.columns:
            sd = df[df["persona"] == seg]
        else:
            sd = df.sample(0)
        n   = len(sd) if len(sd) > 0 else "—"
        pb  = round(sd["interest_binary"].mean()*100, 1) if len(sd) > 0 else "—"
        wtp = int(sd["eco_spend_willingness"].median()) if len(sd) > 0 else "—"
        wtp_str = f"₹{wtp:,}" if isinstance(wtp, int) else "—"
        pb_str  = f"{pb}%" if isinstance(pb, float) else "—"
        st.markdown(f"""
        <div style='border-left:5px solid {a["color"]};background:#fafaf8;
                    border-radius:8px;padding:18px 22px;margin:12px 0'>
          <div style='font-size:20px;font-weight:700;color:{a["color"]}'>
            {a["icon"]} {seg}
            <span style='font-size:12px;color:#888;font-weight:400;margin-left:10px'>
              n={n} · {pb_str} buy · WTP {wtp_str}
            </span>
          </div>
          <div style='font-size:15px;font-weight:600;color:#111;margin:8px 0 4px'>{a["headline"]}</div>
          <table style='width:100%;font-size:13px;border-collapse:collapse'>
            <tr><td style='padding:4px 12px 4px 0;color:#888;width:130px;font-weight:500'>Channel</td>
                <td style='padding:4px 0;color:#333'>{a["channel"]}</td></tr>
            <tr><td style='padding:4px 12px 4px 0;color:#888;font-weight:500'>Message</td>
                <td style='padding:4px 0;color:#333;font-style:italic'>"{a["message"]}"</td></tr>
            <tr><td style='padding:4px 12px 4px 0;color:#888;font-weight:500'>Hero products</td>
                <td style='padding:4px 0;color:#333'>{a["product"]}</td></tr>
            <tr><td style='padding:4px 12px 4px 0;color:#888;font-weight:500'>Price anchor</td>
                <td style='padding:4px 0;color:#333'>{a["price"]}</td></tr>
            <tr><td style='padding:4px 12px 4px 0;color:#888;font-weight:500'>CAC priority</td>
                <td style='padding:4px 0;color:{a["color"]};font-weight:600'>{a["cac"]}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Budget Allocation (₹10L launch)")
    c1,c2 = st.columns(2)
    bdata = {"Eco Warrior": 30, "Mindful Parent": 25, "Urban Renter": 22,
             "Eco Skeptic": 15, "Price Pragmatist": 8}
    bdf = pd.DataFrame({
        "Segment": list(bdata.keys()),
        "Budget %": list(bdata.values()),
        "₹ of 10L": [f"₹{v*10000:,}" for v in bdata.values()],
    })
    with c1:
        fig = px.pie(bdf, names="Segment", values="Budget %", hole=0.4,
                     color="Segment",
                     color_discrete_map={k: ACTIONS[k]["color"] for k in bdata},
                     title="Recommended budget split (₹10L launch)")
        fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Conversion Funnel")
    fdata = pd.DataFrame({
        "Stage": ["Survey respondents","Positive intent (Q34≥4)",
                  "High eco-spend (>₹500/mo)","Will subscribe","Eco Warrior segment"],
        "Count": [
            len(df),
            int(df["interest_binary"].sum()),
            int((df["eco_spend_willingness"] > 500).sum()),
            int(df["will_subscribe"].sum()) if "will_subscribe" in df.columns else 0,
            int((df["persona"] == "Eco Warrior").sum()) if "persona" in df.columns else 0,
        ]
    })
    fig = go.Figure(go.Funnel(
        y=fdata["Stage"], x=fdata["Count"],
        textinfo="value+percent initial",
        marker=dict(color=["#1D9E75","#185FA5","#534AB7","#BA7517","#888780"])
    ))
    fig.update_layout(title="Addressable market funnel", margin=dict(t=50,b=10,l=0,r=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Seasonal Campaign Calendar")
    cal = pd.DataFrame({
        "Month":    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Occasion": ["New Year reset","Valentine gifting","Holi gifting kits",
                     "Summer skin refresh","Mother's Day gifting","Mid-year D2C push",
                     "Monsoon laundry season","Independence Day","Navratri gifting",
                     "Diwali (PEAK)","Diwali tail + Black Friday","Christmas hampers"],
        "Segment":  ["Eco Warrior","Mindful Parent","Urban Renter","Urban Renter","Mindful Parent",
                     "Eco Warrior","Urban Renter","All Segments","Mindful Parent",
                     "All Segments","All Segments","Mindful Parent"],
        "Priority": [3,4,3,3,4,3,4,3,4,5,5,4],
    })
    fig = px.bar(cal, x="Month", y="Priority", color="Segment", text="Occasion",
                 title="Seasonal marketing priority (1=low, 5=peak)",
                 color_discrete_map={k: ACTIONS[k]["color"] for k in ACTIONS})
    fig.update_traces(textposition="inside", textfont_size=9)
    fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=400,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — UPLOAD & SCORE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📥 Upload & Score New Data":
    st.title("📥 Upload & Score New Data")
    st.markdown(
        "Upload a new batch of survey respondents. The pipeline auto-preprocesses, "
        "scores every row with buy probability, predicted eco-spend, cluster assignment, "
        "and recommended marketing action — ready to download."
    )

    with st.spinner("Loading trained pipeline…"):
        m = train_all_models()

    st.success("✅ Random Forest · Gradient Boosting · K-Means (k=5) · Ridge Regressor — all loaded")

    # Template download
    st.markdown("---")
    st.markdown("### Step 1 — Download Template")
    raw = pd.read_csv("zeroplastic_dataset.csv")
    drop_cols = ["persona","noise_type","is_outlier","sdb_flag","interest_binary","interest_3class",
                 "is_eco_warrior","will_subscribe","actual_spend_mid","gap_band","skep_band","gap_band"]
    template = pd.DataFrame(columns=[c for c in raw.columns if c not in drop_cols])
    st.download_button("⬇️ Download CSV Template",
                       template.to_csv(index=False).encode("utf-8"),
                       "zeroplastic_template.csv", "text/csv")

    # Upload
    st.markdown("---")
    st.markdown("### Step 2 — Upload or Use Demo Data")
    uploaded  = st.file_uploader("Upload CSV", type=["csv"])
    use_demo  = st.checkbox("Use 50 demo rows from training data", value=True)

    if uploaded:
        try:
            df_new = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_new)} rows, {len(df_new.columns)} columns")
        except Exception as e:
            st.error(str(e)); st.stop()
    elif use_demo:
        df_new = load_data().sample(50, random_state=99).drop(
            columns=[c for c in drop_cols if c in load_data().columns], errors="ignore")
        st.info("Demo mode: 50 sample rows from training data")
    else:
        st.info("Upload a CSV or enable demo mode to continue."); st.stop()

    # Validation
    st.markdown("---")
    st.markdown("### Step 3 — Validation")
    required_check = ["eco_concern_score","greenwashing_skepticism","green_identity",
                      "city_tier","monthly_hh_income"]
    missing_req = [c for c in required_check if c not in df_new.columns]
    if missing_req:
        st.warning(f"Missing columns (will be imputed with defaults): {missing_req}")
    else:
        st.success("All key columns present")
    v1,v2,v3 = st.columns(3)
    v1.metric("Rows",             len(df_new))
    v2.metric("Columns",          len(df_new.columns))
    v3.metric("Avg missing %",    f"{df_new.isnull().mean().mean()*100:.1f}%")

    # Predict
    st.markdown("---")
    st.markdown("### Step 4 — Run Predictions")
    if st.button("▶ Score All Rows", type="primary"):
        with st.spinner("Preprocessing and scoring…"):
            df_proc   = engineer(df_new)
            feat_cols = m["feat_cols"]
            for c in feat_cols:
                if c not in df_proc.columns:
                    df_proc[c] = np.nan

            X_new = m["pre"].transform(df_proc[feat_cols])

            best_clf = list(m["clf_models"].values())[0]
            df_proc["buy_probability_%"]  = (best_clf.predict_proba(X_new)[:,1] * 100).round(1)
            df_proc["buy_prediction"]     = best_clf.predict(X_new)
            df_proc["predicted_eco_spend_inr"] = m["rfr"].predict(X_new).round(0).astype(int)

            for c in m["clust_feats"]:
                if c not in df_proc.columns:
                    df_proc[c] = 0
            Xc_new = m["km_sc"].transform(df_proc[m["clust_feats"]])
            CNAMES = {
                0: "Eco Warrior",
                1: "Mindful Parent",
                2: "Urban Renter",
                3: "Eco Skeptic",
                4: "Price Pragmatist",
            }
            CACTIONS = {
                "Eco Warrior":      "🌿 Full Home Kit ₹899 + monthly subscription offer",
                "Mindful Parent":   "👨‍👩‍👧 Bathroom Starter Kit ₹499 + safety cert messaging",
                "Urban Renter":     "🏙️ Laundry sheets trial ₹199 on Blinkit/Zepto",
                "Eco Skeptic":      "🔬 Free sample kit + lab certificate link",
                "Price Pragmatist": "💰 Dishwash block ₹149 + per-wash cost calculator",
            }
            cids = m["km"].predict(Xc_new)
            df_proc["cluster_name"]     = [CNAMES.get(c, f"Cluster {c}") for c in cids]
            df_proc["marketing_action"] = df_proc["cluster_name"].map(CACTIONS)

        st.success(f"✅ Scored {len(df_proc)} rows")

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Will Buy",      f"{int(df_proc['buy_prediction'].sum())} / {len(df_proc)}")
        k2.metric("Avg Buy Prob",  f"{df_proc['buy_probability_%'].mean():.1f}%")
        k3.metric("Median Eco-Spend", f"₹{int(df_proc['predicted_eco_spend_inr'].median()):,}/mo")
        k4.metric("Top Cluster",   df_proc["cluster_name"].mode()[0])

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_proc, x="buy_probability_%", nbins=20,
                                color_discrete_sequence=["#1D9E75"],
                                title="Buy probability distribution",
                                labels={"buy_probability_%":"Buy Probability (%)"})
            fig.add_vline(x=50, line_dash="dash", line_color="#A32D2D", annotation_text="50%")
            fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cc = df_proc["cluster_name"].value_counts().reset_index()
            cc.columns = ["Cluster","Count"]
            fig = px.pie(cc, names="Cluster", values="Count", hole=0.4,
                          color_discrete_sequence=CLUSTER_PAL, title="Cluster distribution")
            fig.update_layout(margin=dict(t=50,b=10,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Scored Results")
        out_cols = (["respondent_id"] if "respondent_id" in df_proc.columns else []) + [
            "age_group","gender","city_tier","monthly_hh_income",
            "buy_probability_%","buy_prediction","predicted_eco_spend_inr",
            "cluster_name","marketing_action",
        ]
        out_cols = [c for c in out_cols if c in df_proc.columns]
        out = df_proc[out_cols].copy()
        out["buy_prediction"] = out["buy_prediction"].map({1: "✅ Will Buy", 0: "❌ Won't Buy"})
        out["predicted_eco_spend_inr"] = out["predicted_eco_spend_inr"].apply(lambda x: f"₹{x:,}")
        st.dataframe(out.reset_index(drop=True), use_container_width=True, height=420)

        st.markdown("#### Marketing Action Summary")
        ma = df_proc.groupby("marketing_action").agg(
            n=("buy_probability_%","count"),
            avg_prob=("buy_probability_%","mean"),
            avg_spend=("predicted_eco_spend_inr","mean"),
        ).reset_index()
        ma["avg_prob"]  = ma["avg_prob"].round(1)
        ma["avg_spend"] = ma["avg_spend"].round(0).astype(int).apply(lambda x: f"₹{x:,}")
        ma.columns = ["Marketing Action","Respondents","Avg Buy Prob %","Avg Predicted Eco-Spend"]
        st.dataframe(ma, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.download_button("⬇️ Download Scored CSV",
                           out.to_csv(index=False).encode("utf-8"),
                           "zeroplastic_scored.csv", "text/csv")
        st.info("💡 Once you collect 200+ real responses, merge with training data and retrain for live signal.")
