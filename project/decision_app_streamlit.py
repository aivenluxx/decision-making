
import streamlit as st
from dataclasses import dataclass
import pandas as pd

@dataclass
class Trapezoid:
    a: float; b: float; c: float; d: float
    def centroid(self): return (self.a + 2*self.b + 2*self.c + self.d) / 6.0

LINGUISTIC_SCALE = {
    "VL": Trapezoid(0.0, 0.0, 0.05, 0.2),
    "L":  Trapezoid(0.05, 0.2, 0.2, 0.35),
    "ML": Trapezoid(0.2, 0.35, 0.35, 0.5),
    "M":  Trapezoid(0.35, 0.5, 0.5, 0.65),
    "MH": Trapezoid(0.5, 0.65, 0.65, 0.8),
    "H":  Trapezoid(0.65, 0.8, 0.8, 0.95),
    "VH": Trapezoid(0.8, 0.95, 1.0, 1.0)
}

def fuzzy_weighted_average(traps, weights):
    from dataclasses import dataclass
    agg = Trapezoid(0,0,0,0)
    for t,w in zip(traps, weights):
        agg = Trapezoid(agg.a + t.a*w, agg.b + t.b*w, agg.c + t.c*w, agg.d + t.d*w)
    return agg

def defuzz_pess(tr): return tr.a
def defuzz_opt(tr): return tr.d
def defuzz_neut(tr): return tr.centroid()

st.title("THE METHOD OF AGGREGATION OF TRAPEZOIDAL LINGUISTIC TERMS")
st.write("Upload a CSV with scores or use the example. CSV format: rows - alternatives, columns - criteria; values ​​- term codes (VL,L,ML,M,MH,H,VH).")

uploaded = st.file_uploader("Upload CSV with ratings", type=["csv"])

if uploaded is not None:
    df_ratings = pd.read_csv(uploaded, index_col=0)
    st.write("Ratings:"); st.dataframe(df_ratings)
else:
    st.write("Demonstration data is used.")
    alts = ["A1","A2","A3","A4","A5"]
    crits = ["C1","C2","C3","C4","C5","C6","C7"]
    data = {
        "A1":["MH","M","H","ML","M","H","MH"],
        "A2":["M","ML","MH","M","MH","MH","M"],
        "A3":["L","L","M","L","ML","M","L"],
        "A4":["VH","H","VH","MH","H","VH","VH"],
        "A5":["ML","M","ML","ML","M","ML","ML"]
    }
    df_ratings = pd.DataFrame(data, index=crits).T
    st.dataframe(df_ratings)

# Weights input
st.sidebar.header("Weights (sum should be 1)")
criteria = list(df_ratings.columns)
weights = {}
for c in criteria:
    weights[c] = st.sidebar.number_input(f"Weight {c}", min_value=0.0, max_value=1.0, value=round(1.0/len(criteria),3), step=0.01)

# Normalize weights (if not summing to 1)
ws = list(weights.values())
if sum(ws) == 0:
    norm_ws = [1/len(ws)]*len(ws)
else:
    norm_ws = [w/sum(ws) for w in ws]

st.sidebar.write("Normalized weights:")
st.sidebar.write(dict(zip(criteria, [round(v,3) for v in norm_ws])))

# Compute aggregation
results = []
for alt in df_ratings.index:
    traps = [LINGUISTIC_SCALE[df_ratings.loc[alt,c]] for c in criteria]
    agg = fuzzy_weighted_average(traps, norm_ws)
    results.append({
        "Alternative": alt,
        "a": agg.a, "b": agg.b, "c": agg.c, "d": agg.d,
        "Pessimistic": defuzz_pess(agg),
        "Optimistic": defuzz_opt(agg),
        "Neutral": defuzz_neut(agg)
    })
res_df = pd.DataFrame(results).set_index("Alternative")
res_df["Pess_rank"] = res_df["Pessimistic"].rank(ascending=False, method="min").astype(int)
res_df["Opt_rank"] = res_df["Optimistic"].rank(ascending=False, method="min").astype(int)
res_df["Neut_rank"] = res_df["Neutral"].rank(ascending=False, method="min").astype(int)

st.write("Results:")
st.dataframe(res_df)

st.download_button("Download results CSV", res_df.to_csv().encode('utf-8'), "results_streamlit.csv", "text/csv")
