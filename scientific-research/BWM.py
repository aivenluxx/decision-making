import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="BWM Decision Making", layout="wide")

def solve_linear_bwm(best_idx, worst_idx, bo_vec, ow_vec, n_criteria):
    # Coefficients for the objective function: minimize xi (the last variable)
    c = np.zeros(n_criteria + 1)
    c[-1] = 1

    A_ub = []
    b_ub = []

    # Constraints for Best-to-Others: |w_B - a_Bj * w_j| <= xi
    for j in range(n_criteria):
        if j == best_idx:
            continue
        
        a_Bj = bo_vec[j]
        
        # 1) w_B - a_Bj * w_j - xi <= 0
        row1 = np.zeros(n_criteria + 1)
        row1[best_idx] = 1
        row1[j] = -a_Bj
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)

        # 2) -w_B + a_Bj * w_j - xi <= 0
        row2 = np.zeros(n_criteria + 1)
        row2[best_idx] = -1
        row2[j] = a_Bj
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    # Constraints for Others-to-Worst: |w_j - a_jW * w_W| <= xi
    for j in range(n_criteria):
        if j == worst_idx:
            continue
        
        a_jW = ow_vec[j]
        
        # 1) w_j - a_jW * w_W - xi <= 0
        row1 = np.zeros(n_criteria + 1)
        row1[j] = 1
        row1[worst_idx] = -a_jW
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)
        
        # 2) -w_j + a_jW * w_W - xi <= 0
        row2 = np.zeros(n_criteria + 1)
        row2[j] = -1
        row2[worst_idx] = a_jW
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    # Sum of weights constraint: sum(w) = 1
    A_eq = [np.ones(n_criteria + 1)]
    A_eq[0][-1] = 0
    b_eq = [1]

    # Bounds: w >= 0, xi >= 0
    bounds = [(0, None) for _ in range(n_criteria + 1)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        weights = res.x[:-1] 
        consistency_ratio = res.x[-1]
        return weights, consistency_ratio
    else:
        return None, None

def main():
    st.title("Best-Worst Method (BWM) Calculator")

    st.sidebar.header("Settings")
    
    num_alternatives = st.sidebar.number_input("Quantity of alternatives", min_value=2, value=6, step=1)
    num_criteria = st.sidebar.number_input("Quantity of criteria", min_value=2, value=7, step=1)
    
    st.sidebar.header("Custom Names")

    with st.sidebar.expander("Edit Criteria Names"):
        if 'criteria_names_custom' not in st.session_state or len(st.session_state['criteria_names_custom']) != num_criteria:
            st.session_state['criteria_names_custom'] = [f"Criterion {i+1}" for i in range(num_criteria)]

        new_criteria_names = []
        st.write("Enter names for the criteria:")
        for i in range(num_criteria):
            name = st.text_input(f"Criterion {i+1} Name", 
                                 value=st.session_state['criteria_names_custom'][i], 
                                 key=f"crit_name_{i}")
            new_criteria_names.append(name)
        
        criteria_names = new_criteria_names
        st.session_state['criteria_names_custom'] = new_criteria_names

    with st.sidebar.expander("Edit Alternative Names"):
        if 'alternatives_names_custom' not in st.session_state or len(st.session_state['alternatives_names_custom']) != num_alternatives:
            st.session_state['alternatives_names_custom'] = [f"Alternative {i+1}" for i in range(num_alternatives)]

        new_alternatives_names = []
        st.write("Enter names for the alternatives:")
        for i in range(num_alternatives):
            name = st.text_input(f"Alternative {i+1} Name", 
                                 value=st.session_state['alternatives_names_custom'][i], 
                                 key=f"alt_name_{i}")
            new_alternatives_names.append(name)
        
        alternatives_names = new_alternatives_names
        st.session_state['alternatives_names_custom'] = new_alternatives_names
    
    st.divider()

    st.header("1. Defining Best and Worst Criteria")
    
    col1, col2 = st.columns(2)
    with col1:
        best_criterion = st.selectbox("Choose the **BEST** Criterion:", criteria_names, index=0)
    with col2:
        default_worst_idx = min(len(criteria_names) - 1, len(criteria_names) - 1)
        worst_criterion = st.selectbox("Choose the **WORST** Criterion:", criteria_names, index=default_worst_idx)

    if best_criterion == worst_criterion:
        st.warning("Warning: The Best and Worst criteria are the same. This is only meaningful if all criteria are equally important.")

    best_idx = criteria_names.index(best_criterion)
    worst_idx = criteria_names.index(worst_criterion)

    st.divider()

    st.header("2. Pairwise Comparison (Scale 1-9)")
    st.info("1 - Equal importance, 9 - Absolute advantage (Best-to-Others) or Absolute unimportance (Others-to-Worst)")

    st.subheader(f"Comparison of the Best ({best_criterion}) with Others (Best-to-Others)")
    bo_vec = np.ones(num_criteria)
    
    cols_bo = st.columns(3)
    for i in range(num_criteria):
        if i == best_idx:
            continue
        with cols_bo[i % 3]:
            val = st.number_input(
                f"Preference of {best_criterion} over {criteria_names[i]}:", 
                min_value=1, max_value=9, value=1, key=f"bo_{i}"
            )
            bo_vec[i] = val

    st.subheader(f"Comparison of Others to the Worst ({worst_criterion}) (Others-to-Worst)")
    ow_vec = np.ones(num_criteria)
    
    cols_ow = st.columns(3)
    for i in range(num_criteria):
        if i == worst_idx:
            continue 
        with cols_ow[i % 3]:
            val = st.number_input(
                f"Preference of {criteria_names[i]} over {worst_criterion}:", 
                min_value=1, max_value=9, value=1, key=f"ow_{i}"
            )
            ow_vec[i] = val

    st.divider()

    if st.button("Calculate Criteria Weights"):
        weights, xi = solve_linear_bwm(best_idx, worst_idx, bo_vec, ow_vec, num_criteria)
        
        if weights is not None:
            st.session_state['weights'] = weights
            st.session_state['xi'] = xi
            st.session_state['calculated'] = True
        else:
            st.error("Could not find an optimal solution. Please check your input data.")

    if st.session_state.get('calculated', False):
        weights = st.session_state['weights']
        xi = st.session_state['xi']

        st.success(f"Calculation Successful! Consistency Ratio (ξ): {xi:.4f}")

        # --- Visualizing Weights ---
        weights_df = pd.DataFrame({
            "Criterion": criteria_names,
            "Weight": weights
        }).sort_values(by="Weight", ascending=False)

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.bar_chart(weights_df.set_index("Criterion"))
        with col_table:
            st.dataframe(weights_df.style.format({"Weight": "{:.4f}"}))

        # --- NEW: Consistency Error Analysis ---
        st.subheader("Consistency Error Analysis")
        st.caption("This table shows the absolute deviation (error) for each comparison you made. The maximum error corresponds to ξ.")
        
        w_best = weights[best_idx]
        w_worst = weights[worst_idx]
        
        error_data = []
        
        # Calculate Best-to-Others Errors: |w_B - a_Bj * w_j|
        for i in range(num_criteria):
            if i != best_idx:
                a_Bj = bo_vec[i]
                w_j = weights[i]
                error = abs(w_best - a_Bj * w_j)
                error_data.append({
                    "Type": "Best-to-Others",
                    "Pair": f"{best_criterion} > {criteria_names[i]}",
                    "Input Value (a)": a_Bj,
                    "Calculated Ratio (w_B/w_j)": w_best/w_j if w_j != 0 else 0,
                    "Error (Abs Deviation)": error
                })

        # Calculate Others-to-Worst Errors: |w_j - a_jW * w_W|
        for i in range(num_criteria):
            if i != worst_idx:
                a_jW = ow_vec[i]
                w_j = weights[i]
                error = abs(w_j - a_jW * w_worst)
                error_data.append({
                    "Type": "Others-to-Worst",
                    "Pair": f"{criteria_names[i]} > {worst_criterion}",
                    "Input Value (a)": a_jW,
                    "Calculated Ratio (w_j/w_W)": w_j/w_worst if w_worst != 0 else 0,
                    "Error (Abs Deviation)": error
                })
        
        df_errors = pd.DataFrame(error_data).sort_values(by="Error (Abs Deviation)", ascending=False)
        st.dataframe(df_errors.style.format({
            "Calculated Ratio (w_B/w_j)": "{:.4f}",
            "Calculated Ratio (w_j/w_W)": "{:.4f}",
            "Error (Abs Deviation)": "{:.4f}"
        }))

        st.divider()

        st.header("3. Evaluation and Ranking of Alternatives")
        st.write("Enter the scores for each alternative against each criterion.")
        
        if 'df_alternatives' not in st.session_state or \
           st.session_state['df_alternatives'].shape != (num_alternatives, num_criteria) or \
           list(st.session_state['df_alternatives'].columns) != criteria_names or \
           list(st.session_state['df_alternatives'].index) != alternatives_names:
            
            data_init = np.random.randint(1, 10, size=(num_alternatives, num_criteria))
            st.session_state['df_alternatives'] = pd.DataFrame(
                data_init,
                columns=criteria_names,
                index=alternatives_names
            )
        
        edited_df = st.data_editor(st.session_state['df_alternatives'])

        st.write("Set optimization direction for criteria (max - higher is better; min - lower is better):")
        opt_dirs = []
        cols_opt = st.columns(4)
        for i, crit in enumerate(criteria_names):
            with cols_opt[i % 4]:
                opt = st.selectbox(f"{crit}", ["max", "min"], key=f"opt_{i}")
                opt_dirs.append(opt)

        if st.button("Rank Alternatives"):

            matrix = edited_df.values.astype(float)
            norm_matrix = np.zeros_like(matrix)

            for j in range(num_criteria):
                col = matrix[:, j]
                if opt_dirs[j] == "max":
                    col_max = col.max()
                    norm_matrix[:, j] = col / col_max if col_max != 0 else 0
                else:
                    min_val = col.min()
                    norm_matrix[:, j] = np.array([min_val / x if x != 0 else 0 for x in col])

            final_scores = np.dot(norm_matrix, weights)

            results_df = pd.DataFrame({
                "Alternative": alternatives_names,
                "Final Score": final_scores
            }).sort_values(by="Final Score", ascending=False)
            
            results_df["Rank"] = range(1, len(results_df) + 1)

            st.subheader("Ranking Results")
            st.dataframe(results_df.style.format({"Final Score": "{:.4f}"}))
            
            st.bar_chart(results_df.set_index("Alternative")["Final Score"])

if __name__ == "__main__":
    main()