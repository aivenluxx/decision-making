import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Trapezoid:
    a: float; b: float; c: float; d: float
    def centroid(self): return (self.a + 2*self.b + 2*self.c + self.d) / 6.0

@dataclass
class Triangle:
    a: float; b: float; c: float
    def centroid(self): return (self.a + self.b + self.c) / 3.0



LINGUISTIC_SCALE = {
    "VL": Trapezoid(0.0, 0.0, 0.05, 0.2),
    "L":  Triangle(0.05, 0.2, 0.35),
    "ML": Triangle(0.2, 0.35, 0.5),
    "M":  Triangle(0.35, 0.5, 0.65),
    "MH": Triangle(0.5, 0.65, 0.8),
    "H":  Triangle(0.65, 0.8, 0.95),
    "VH": Trapezoid(0.8, 0.95, 1.0, 1.0)
}


def fuzzy_weighted_average(traps, weights):
    agg_a = agg_b = agg_c = agg_d = 0
    for t, w in zip(traps, weights):
        if isinstance(t, Triangle):
            agg_a += t.a * w
            agg_b += t.b * w
            agg_c += t.c * w
            agg_d += t.c * w
        else:
            agg_a += t.a * w
            agg_b += t.b * w
            agg_c += t.c * w
            agg_d += t.d * w
    return Trapezoid(agg_a, agg_b, agg_c, agg_d)

def defuzz_pess(tr): return tr.a
def defuzz_opt(tr): return tr.d
def defuzz_neut(tr): return tr.centroid()



def plot_term(term, name="Term", color="blue"):
    plt.figure(figsize=(4, 2))
    if isinstance(term, Triangle):
        x = [term.a, term.b, term.c]; y = [0, 1, 0]
    else:
        x = [term.a, term.b, term.c, term.d]; y = [0, 1, 1, 0]
    plt.plot(x, y, color=color, label=name)
    plt.fill_between(x, y, alpha=0.3, color=color)
    plt.title(f"{name} ({'Triangle' if isinstance(term, Triangle) else 'Trapezoid'})")
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 1)
    plt.xlabel("x"); plt.ylabel("Membership degree")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

def plot_all_terms(terms_dict):
    plt.figure(figsize=(5, 3))
    for name, term in terms_dict.items():
        if isinstance(term, Triangle):
            x = [term.a, term.b, term.c]; y = [0, 1, 0]
        else:
            x = [term.a, term.b, term.c, term.d]; y = [0, 1, 1, 0]
        plt.plot(x, y, label=name)
        plt.fill_between(x, y, alpha=0.2)
    plt.title("Summary of Linguistic Terms")
    plt.xlabel("x"); plt.ylabel("Membership degree")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    st.sidebar.pyplot(plt.gcf())
    plt.close()



st.title("Aggregation of Trapezoidal and Triangular Linguistic Terms")

uploaded = st.file_uploader("Upload CSV with ratings", type=["csv"])
if uploaded is not None:
    df_ratings = pd.read_csv(uploaded, index_col=0)
else:
    st.write("Using demonstration data")
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



st.sidebar.header("Manage Criteria Weights")

if "weights" not in st.session_state:
    st.session_state.weights = {c: round(1.0/len(df_ratings.columns), 3) for c in df_ratings.columns}

new_crit = st.sidebar.text_input("Add new criterion name")
new_weight = st.sidebar.number_input("New criterion weight", 0.0, 1.0, 0.1, 0.01)
if st.sidebar.button("Add criterion") and new_crit:
    st.session_state.weights[new_crit] = new_weight
    st.sidebar.success(f"Added criterion {new_crit}")

remove_crit = st.sidebar.selectbox("Remove criterion", ["None"] + list(st.session_state.weights.keys()))
if remove_crit != "None" and st.sidebar.button("Delete selected criterion"):
    del st.session_state.weights[remove_crit]
    st.sidebar.warning(f"Deleted criterion {remove_crit}")

weights = {}
for c, val in st.session_state.weights.items():
    weights[c] = st.sidebar.number_input(f"Weight {c}", 0.0, 1.0, val, 0.01)
st.session_state.weights = weights

ws = list(weights.values())
norm_ws = [w/sum(ws) for w in ws] if sum(ws)>0 else [1/len(ws)]*len(ws)
st.sidebar.write("Normalized weights:", dict(zip(weights.keys(), [round(v,3) for v in norm_ws])))



st.sidebar.header("Manage Linguistic Terms")

term_to_remove = st.sidebar.selectbox("Select term to delete", ["None"] + list(LINGUISTIC_SCALE.keys()))
if term_to_remove != "None" and st.sidebar.button("Delete selected term"):
    del LINGUISTIC_SCALE[term_to_remove]
    st.sidebar.warning(f"Deleted term {term_to_remove}")

for term, shape in list(LINGUISTIC_SCALE.items()):
    with st.sidebar.expander(f"Edit {term}"):
        if isinstance(shape, Triangle):
            a = st.number_input(f"{term} a", value=shape.a, step=0.01, key=f"a_{term}")
            b = st.number_input(f"{term} b", value=shape.b, step=0.01, key=f"b_{term}")
            c = st.number_input(f"{term} c", value=shape.c, step=0.01, key=f"c_{term}")
            LINGUISTIC_SCALE[term] = Triangle(a, b, c)
        else:
            a = st.number_input(f"{term} a", value=shape.a, step=0.01, key=f"a_{term}")
            b = st.number_input(f"{term} b", value=shape.b, step=0.01, key=f"b_{term}")
            c = st.number_input(f"{term} c", value=shape.c, step=0.01, key=f"c_{term}")
            d = st.number_input(f"{term} d", value=shape.d, step=0.01, key=f"d_{term}")
            LINGUISTIC_SCALE[term] = Trapezoid(a,b,c,d)
        plot_term(LINGUISTIC_SCALE[term], name=term)

plot_all_terms(LINGUISTIC_SCALE)



st.subheader("Intermediate Calculations")

intermediate_data = []
criteria = list(weights.keys())

for alt in df_ratings.index:
    traps = [LINGUISTIC_SCALE[df_ratings.loc[alt, c]] for c in criteria if c in df_ratings.columns]
    for c, t, w in zip(criteria, traps, norm_ws[:len(traps)]):
        if isinstance(t, Triangle):
            a, b, c_, d = t.a, t.b, t.c, t.c
        else:
            a, b, c_, d = t.a, t.b, t.c, t.d
        intermediate_data.append({
            "Alternative": alt,
            "Criterion": c,
            "Term": df_ratings.loc[alt, c],
            "a": round(a,3), "b": round(b,3), "c": round(c_,3), "d": round(d,3),
            "Weight": round(w,3),
            "Weighted a": round(a*w,3), "Weighted b": round(b*w,3),
            "Weighted c": round(c_*w,3), "Weighted d": round(d*w,3)
        })

inter_df = pd.DataFrame(intermediate_data)
st.dataframe(inter_df)
st.download_button(
    "Download intermediate calculations CSV",
    inter_df.to_csv(index=False).encode('utf-8'),
    "intermediate_calculations.csv", "text/csv"
)



results = []
for alt in df_ratings.index:
    traps = [LINGUISTIC_SCALE[df_ratings.loc[alt, c]] for c in criteria if c in df_ratings.columns]
    agg = fuzzy_weighted_average(traps, norm_ws[:len(traps)])
    results.append({
        "Alternative": alt,
        "a": round(agg.a,4), "b": round(agg.b,4), "c": round(agg.c,4), "d": round(agg.d,4),
        "Pessimistic": round(defuzz_pess(agg),4),
        "Optimistic": round(defuzz_opt(agg),4),
        "Neutral": round(defuzz_neut(agg),4)
    })

interval_df = pd.DataFrame(results).set_index("Alternative")

st.subheader("Formed Interval Estimates (Aggregated Fuzzy Numbers)")
st.dataframe(interval_df[["a","b","c","d"]])



st.subheader("Defuzzification: Decision Maker’s Positions")

interval_df["Pess_rank"] = interval_df["Pessimistic"].rank(ascending=False, method="min").astype(int)
interval_df["Opt_rank"] = interval_df["Optimistic"].rank(ascending=False, method="min").astype(int)
interval_df["Neut_rank"] = interval_df["Neutral"].rank(ascending=False, method="min").astype(int)

st.write("**Pessimistic, Optimistic, and Neutral Evaluations with Ranking:**")
st.dataframe(interval_df[["Pessimistic","Optimistic","Neutral","Pess_rank","Opt_rank","Neut_rank"]])

st.download_button(
    "Download results CSV",
    interval_df.to_csv().encode('utf-8'),
    "results_streamlit.csv", "text/csv"
)
