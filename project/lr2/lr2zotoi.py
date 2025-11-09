import streamlit as st
import pandas as pd
import numpy as np
import random
from copy import deepcopy

st.set_page_config(page_title="Fuzzy ARAS (Doc)", layout="wide")

def interval_fuzzy(l, lp, m, up, u):
    return np.array([float(l), float(lp), float(m), float(up), float(u)])

def fuzzy_add(a, b):
    return a + b 

def fuzzy_mul(a, b):
    return a * b

def interval_str(t):
    return f"[{t[0]:.4f}; {t[1]:.4f}; {t[2]:.4f}; {t[3]:.4f}; {t[4]:.4f}]"


def fuzzy_invert_and_normalize(fuzzy_num, a_minus_sum):
        
    safe_l = np.where(fuzzy_num[0] == 0, 1e-9, fuzzy_num[0])
    safe_lp = np.where(fuzzy_num[1] == 0, 1e-9, fuzzy_num[1])
    safe_m = np.where(fuzzy_num[2] == 0, 1e-9, fuzzy_num[2])
    safe_up = np.where(fuzzy_num[3] == 0, 1e-9, fuzzy_num[3])
    safe_u = np.where(fuzzy_num[4] == 0, 1e-9, fuzzy_num[4])

    
    inverted_num = np.array([
        1 / safe_u,
        1 / safe_up,
        1 / safe_m,
        1 / safe_lp,
        1 / safe_l
    ])
    
    
    safe_a_minus_sum = np.where(a_minus_sum == 0, 1e-9, a_minus_sum)
    return inverted_num / safe_a_minus_sum


def fuzzy_div_norm(x, c_sum):
   
    safe_c_sum = np.where(c_sum == 0, 1e-9, c_sum)
    return x / safe_c_sum

def expert_aggregation(list_of_tri):
    if not list_of_tri:
        return interval_fuzzy(0, 0, 0, 0, 0)
        
    K = len(list_of_tri)
    arr = np.array(list_of_tri) 
    
    if K == 1:
        l = arr[0, 0]
        lp = arr[0, 0]
        m = arr[0, 1]
        up = arr[0, 2]
        u = arr[0, 2]
        return interval_fuzzy(l, lp, m, up, u)
    
    l = np.min(arr[:, 0])
    
    if np.any(arr[:, 0] == 0):
        lp = 0.0
    else:
        l_prod = np.prod(arr[:, 0])
        lp = l_prod**(1/K)
    
    if np.any(arr[:, 1] == 0):
        m = 0.0
    else:
        m_prod = np.prod(arr[:, 1])
        m = m_prod**(1/K)

    if np.any(arr[:, 2] == 0):
        up = 0.0
    else:
        up_prod = np.prod(arr[:, 2])
        up = up_prod**(1/K)
    
    u = np.max(arr[:, 2])

    sorted_vals = np.sort([l, lp, m, up, u])
    return interval_fuzzy(sorted_vals[0], sorted_vals[1], sorted_vals[2], sorted_vals[3], sorted_vals[4])

def defuzzification(five_val_fuzzy):
    return np.sum(five_val_fuzzy) / 5



LINGUISTIC_MAP = {
    "VP": np.array([0.0, 0.0, 0.1]),
    "P":  np.array([0.0, 0.1, 0.3]),
    "MP": np.array([0.1, 0.3, 0.5]),
    "F":  np.array([0.3, 0.5, 0.7]),
    "MG": np.array([0.5, 0.7, 0.9]),
    "G":  np.array([0.7, 0.7, 1.0]),
    "VG": np.array([0.9, 1.0, 1.0])
}
WEIGHT_LINGUISTIC_MAP = {
    "VL": np.array([0.0, 0.0, 0.1]),
    "L":  np.array([0.0, 0.1, 0.3]),
    "ML": np.array([0.1, 0.3, 0.5]),
    "M":  np.array([0.3, 0.5, 0.7]),
    "MH": np.array([0.5, 0.7, 0.9]),
    "H":  np.array([0.7, 0.7, 1.0]),
    "VH": np.array([0.9, 1.0, 1.0])
}

def tri_str(t):
    return f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"

st.title("Fuzzy ARAS (Doc-compliant implementation)")

with st.sidebar:
    st.header("Setting of task")
    num_alts = st.number_input("Quantity of alternatives (>=4)", min_value=4, max_value=20, value=4, step=1)
    num_criteria = st.number_input("Quantity of criterias (>=5)", min_value=5, max_value=20, value=5, step=1)
    num_experts = 4
    st.write(f"Quantity of experts (fixed by doc): **{num_experts}**")

    st.markdown("---")
    st.markdown("Linguistic Terms Map (Alternatives):")
    st.json({k: tri_str(v) for k, v in LINGUISTIC_MAP.items()})
    st.markdown("Linguistic Terms Map (Weights):")
    st.json({k: tri_str(v) for k, v in WEIGHT_LINGUISTIC_MAP.items()})

alts = [f"A{i+1}" for i in range(int(num_alts))]
criteria = [f"C{j+1}" for j in range(int(num_criteria))]
experts = [f"E{k+1}" for k in range(int(num_experts))]

st.subheader("Criterias and weights")
col1, col2 = st.columns([2,3])

with col1:
    st.write("Define type of criteria (Benefit / Cost).")
    crit_types = []
    for c in criteria:
        t = st.selectbox(f"Type {c}", options=["benefit", "cost"], index=0, key=f"type_{c}")
        crit_types.append(t)
with col2:
    st.write("Enter weights of LT (for criteria)")
    crit_weights = {}
    for c in criteria:
        term = st.selectbox(f"Weight {c}", options=list(WEIGHT_LINGUISTIC_MAP.keys()), 
                            index=list(WEIGHT_LINGUISTIC_MAP.keys()).index("M") if "M" in WEIGHT_LINGUISTIC_MAP else 3, 
                            key=f"w_{c}")
        crit_weights[c] = WEIGHT_LINGUISTIC_MAP[term]

st.subheader("Expert assesments (Linguistic Terms for Alternatives)")

def get_default_assessments(alts_list, criteria_list, experts_list):
    return {e: {a: {c: "F" for c in criteria_list} for a in alts_list} for e in experts_list}

current_dims = (tuple(alts), tuple(criteria), tuple(experts))
if 'experts_data' not in st.session_state or st.session_state.get('assessment_dims') != current_dims:
    st.session_state.experts_data = get_default_assessments(alts, criteria, experts)
    st.session_state.assessment_dims = current_dims
    
if st.sidebar.button("🎲 Generate random expert assessments"):
    for e in experts:
        for a in alts:
            for c in criteria:
                term = random.choice(list(LINGUISTIC_MAP.keys()))
                st.session_state.experts_data[e][a][c] = term
    st.success("Random assessments generated!")

st.markdown("Ви можете **завантажити CSV** з оцінками або **ввести їх вручну** нижче.")
st.markdown("Формат CSV: `Expert`, `Alternative`, `Criterion`, `Term` (наприклад, `E1,A1,C1,VG`)")

uploaded_file = st.file_uploader("Upload Expert Assessments (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        required_cols = {"Expert", "Alternative", "Criterion", "Term"}
        if not required_cols.issubset(df.columns):
            st.error(f"Error: CSV must contain columns: {', '.join(required_cols)}")
        else:
            loaded_count = 0
            for idx, row in df.iterrows():
                e = str(row["Expert"])
                a = str(row["Alternative"])
                c = str(row["Criterion"])
                term = str(row["Term"]).strip().upper() 
                
                if e in experts and a in alts and c in criteria and term in LINGUISTIC_MAP:
                    st.session_state.experts_data[e][a][c] = term
                    loaded_count += 1
                else:
                    st.warning(f"Skipping row {idx}: Invalid data '{e}', '{a}', '{c}', or '{term}'")
            
            st.success(f"Successfully loaded and validated {loaded_count} assessments from CSV.")
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

else:
    st.markdown("---")
    st.markdown("#### Manual Input")
    for e in experts:
        st.markdown(f"**Expert {e}**")
        for a in alts:
            cols = st.columns(int(num_criteria))
            for i, c in enumerate(criteria):
                with cols[i]:
                    term_key = f"inp_{e}_{a}_{c}"
                    term_options = list(LINGUISTIC_MAP.keys())
                    
                    current_term = st.session_state.experts_data[e][a][c]
                    if current_term not in term_options:
                        current_term = "F" 
                        
                    current_index = term_options.index(current_term)
                    
                    new_term = st.selectbox(
                        f"{a}-{c}", 
                        options=term_options, 
                        index=current_index, 
                        key=term_key
                    )
                    st.session_state.experts_data[e][a][c] = new_term

if st.button("Go"):
    
    experts_data_calc = {}
    for e, alt_map in st.session_state.experts_data.items():
        experts_data_calc[e] = {}
        for a, crit_map in alt_map.items():
            experts_data_calc[e][a] = {}
            for c, term in crit_map.items():
                experts_data_calc[e][a][c] = LINGUISTIC_MAP[term]
    
    
    
    st.header("1. Aggregation of Expert Assessments")
    agg_matrix = {a: {} for a in alts}
    agg_weights = {} 
    
    for c in criteria:
        tris = [crit_weights[c] for _ in range(num_experts)]
        agg_weights[c] = expert_aggregation(tris)
    
    st.markdown("#### Aggregated Weights (W_j)")
    w_df = pd.DataFrame([{"Criterion": c, "W_j": interval_str(agg_weights[c])} for c in criteria]).set_index("Criterion")
    st.dataframe(w_df)
    
    for a in alts:
        for c in criteria:
            tris = [experts_data_calc[e][a][c] for e in experts]
            agg_matrix[a][c] = expert_aggregation(tris)

    st.markdown("#### Aggregated Alternatives Matrix (X_ij)")
    agg_df = pd.DataFrame({c: {a: interval_str(agg_matrix[a][c]) for a in alts} for c in criteria})
    st.dataframe(agg_df)


    
    st.header("2. Optimal Value Matrix (X_0j) & Denominators")
    x0_matrix = {}
    
   
    c_plus_sum = {}  
    a_minus_sum = {} 
    
    for j, c in enumerate(criteria):
        vals = np.array([agg_matrix[a][c] for a in alts]) 
        
        if crit_types[j] == "benefit": 
            
            x0_l = np.max(vals[:, 0]) 
            x0_lp = np.max(vals[:, 1])
            x0_m = np.max(vals[:, 2])
            x0_up = np.max(vals[:, 3])
            x0_u = np.max(vals[:, 4])
            x0_matrix[c] = interval_fuzzy(x0_l, x0_lp, x0_m, x0_up, x0_u)
        else: 
           
            x0_l = np.min(vals[:, 0]) 
            x0_lp = np.min(vals[:, 1])
            x0_m = np.min(vals[:, 2])
            x0_up = np.min(vals[:, 3])
            x0_u = np.min(vals[:, 4])
            x0_matrix[c] = interval_fuzzy(x0_l, x0_lp, x0_m, x0_up, x0_u)
            
        
        c_alt_sum = np.sum(vals[:, 4]) 
        c0j = x0_matrix[c][4]          
        c_plus_sum[c] = c_alt_sum + c0j
        
        
        l_values = vals[:, 0] 
        l0_value = x0_matrix[c][0] 
        
        
        safe_l_values = np.where(l_values == 0, 1e-9, l_values)
        safe_l0_value = np.where(l0_value == 0, 1e-9, l0_value)
        
        inv_l_sum = np.sum(1 / safe_l_values) 
        inv_l0 = 1 / safe_l0_value           
        a_minus_sum[c] = inv_l_sum + inv_l0
        
    
    x0_df_data = [{
        "Criterion": c, 
        "X_0j": interval_str(x0_matrix[c]), 
        "c_j+ (Benefit Denom.)": f"{c_plus_sum[c]:.4f}",
        "a_j- (Cost Denom.)": f"{a_minus_sum[c]:.4f}"
    } for c in criteria]
    
    x0_df = pd.DataFrame(x0_df_data).set_index("Criterion")
    st.dataframe(x0_df)
    
    
    
    st.header("3. Normalized Matrix (R_ij)")
    norm_matrix = {a: {} for a in alts}
    norm_optimal = {} 
    
    
    for j, c in enumerate(criteria): 
        if crit_types[j] == "benefit":
            
            c_j_plus = c_plus_sum[c]
            
            
            norm_optimal[c] = fuzzy_div_norm(x0_matrix[c], c_j_plus)
            
            
            for a in alts:
                norm_matrix[a][c] = fuzzy_div_norm(agg_matrix[a][c], c_j_plus)
                
        else:
            
            a_j_minus = a_minus_sum[c]
            
            
            norm_optimal[c] = fuzzy_invert_and_normalize(x0_matrix[c], a_j_minus)

            
            for a in alts:
                norm_matrix[a][c] = fuzzy_invert_and_normalize(agg_matrix[a][c], a_j_minus)
    

    norm_df_data = []
    norm_df_data.append({"Alternative": "Optimal (R_0j)", **{c: interval_str(norm_optimal[c]) for c in criteria}})
    for a in alts:
        norm_df_data.append({"Alternative": a, **{c: interval_str(norm_matrix[a][c]) for c in criteria}})
        
    norm_df = pd.DataFrame(norm_df_data).set_index("Alternative")
    st.dataframe(norm_df)


    
    st.header("4. Normalized Weighted Matrix (V_ij)")
    weighted_matrix = {a: {} for a in alts}
    weighted_optimal = {}
    
    for c in criteria:
        weighted_optimal[c] = fuzzy_mul(norm_optimal[c], agg_weights[c])
        for a in alts:
            weighted_matrix[a][c] = fuzzy_mul(norm_matrix[a][c], agg_weights[c])

    weighted_df_data = []
    weighted_df_data.append({"Alternative": "Optimal (V_0j)", **{c: interval_str(weighted_optimal[c]) for c in criteria}})
    for a in alts:
        weighted_df_data.append({"Alternative": a, **{c: interval_str(weighted_matrix[a][c]) for c in criteria}})
        
    weighted_df = pd.DataFrame(weighted_df_data).set_index("Alternative")
    st.dataframe(weighted_df)


    
    st.header("5. Overall Optimality Score (S_i)")
    
    S_opt = sum((weighted_optimal[c] for c in criteria), np.zeros(5))
    S = {a: sum((weighted_matrix[a][c] for c in criteria), np.zeros(5)) for a in alts}
    
    s_df_data = []
    s_df_data.append({"Alternative": "Optimal", "S_i": interval_str(S_opt)})
    for a in alts:
        s_df_data.append({"Alternative": a, "S_i": interval_str(S[a])})
        
    s_df = pd.DataFrame(s_df_data).set_index("Alternative")
    st.dataframe(s_df)


    
    st.header("6. Defuzzification (S_def)")
    S_def_opt = defuzzification(S_opt)
    S_def = {a: defuzzification(S[a]) for a in alts}

    def_df_data = []
    def_df_data.append({"Alternative": "Optimal (S_def_opt)", "S_def": f"{S_def_opt:.4f}"})
    for a in alts:
        def_df_data.append({"Alternative": a, "S_def": f"{S_def[a]:.4f}"})
        
    def_df = pd.DataFrame(def_df_data).set_index("Alternative")
    st.dataframe(def_df)
    
    
    
    st.header("7. Degree of Utility (Q_i)")
    Q = {a: (S_def[a] / S_def_opt) if S_def_opt != 0 else 0 for a in alts}
    
    Q_df = pd.DataFrame([
        {"Alternative": a, "Q_i": f"{Q[a]:.4f}"} for a in alts
    ]).set_index("Alternative").sort_values(by="Q_i", ascending=False)
    
    st.dataframe(Q_df)

    
    st.header("Final Result")
    if not Q_df.empty:
        best_alt = Q_df.index[0]
        st.balloons()
        st.success(f"Best alternative: **{best_alt}** (Q_i = {Q_df.loc[best_alt,'Q_i']})")
    else:
        st.error("Calculation resulted in empty data.")

else:
    st.info("Customize settings, load CSV or input assessments, and then press 'Go'.")