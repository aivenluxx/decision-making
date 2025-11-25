import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TOPSIS", layout="wide")

st.title("TOPSIS Method Application")

st.sidebar.header("1.Setting of model")

num_alternatives = st.sidebar.number_input(
    "Quanitity of alternatives", 
    min_value=6, value=6, step=1
)
num_criteria = st.sidebar.number_input(
    "Quantity of criterions", 
    min_value=7, value=7, step=1
)

st.sidebar.info(f"Created configuration: {num_alternatives} alternatives, {num_criteria} criterions.")

def run_topsis():
    st.header("2. Entering data")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Names of alternatives")
        alt_names = [st.text_input(f"Alternative {i+1}", value=f"Alt_{i+1}", key=f"alt_{i}") for i in range(num_alternatives)]
    
    with col2:
        st.subheader("Names of criterions")
        crit_names = [st.text_input(f"Criteria {j+1}", value=f"Crit_{j+1}", key=f"crit_{j}") for j in range(num_criteria)]

    st.subheader("Decision Matrix (X)")
    st.write("Enter value for each alternative for each criterion:")
    
    default_data = np.random.randint(1, 10, size=(num_alternatives, num_criteria))
    df_input = pd.DataFrame(default_data, index=alt_names, columns=crit_names)
    
    edited_df = st.data_editor(df_input, use_container_width=True)

    st.subheader("Settings of criterions (weights and type)")
    
    params_df = pd.DataFrame({
        "Criteria": crit_names,
        "Weight (w)": [1.0/num_criteria] * num_criteria, 
        "Type": ["Benefit (Max)"] * num_criteria
    })
    
    edited_params = st.data_editor(
        params_df, 
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Type of optimization",
                options=["Benefit (Max)", "Cost (Min)"],
                help="Benefit: more - better. Cost: less - better."
            ),
            "Weight (w)": st.column_config.NumberColumn(
                "Weight",
                min_value=0.0,
                max_value=1.0,
                format="%.4f"
            )
        },
        hide_index=True,
        use_container_width=True
    )

    total_weight = edited_params["Weight (w)"].sum()
    if not np.isclose(total_weight, 1.0):
        st.warning(f"Sum of weights equal {total_weight:.4f}. Recommended to set sum equal 1.0.")
    
    if st.button("Calculate TOPSIS"):
        
        try:
            matrix = edited_df.values.astype(float)
            weights = edited_params["Weight (w)"].values
            types = edited_params["Type"].values
            
            m, n = matrix.shape
            
            divisors = np.sqrt((matrix**2).sum(axis=0))
            normalized_matrix = matrix / divisors
            
            weighted_matrix = normalized_matrix * weights
            
            ideal_best = []
            ideal_worst = []
            
            for j in range(n):
                if types[j] == "Benefit (Max)":
                    ideal_best.append(np.max(weighted_matrix[:, j]))
                    ideal_worst.append(np.min(weighted_matrix[:, j]))
                else: # Cost
                    ideal_best.append(np.min(weighted_matrix[:, j]))
                    ideal_worst.append(np.max(weighted_matrix[:, j]))
            
            ideal_best = np.array(ideal_best)
            ideal_worst = np.array(ideal_worst)

            dist_pos = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
            dist_neg = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
            
            closeness = dist_neg / (dist_pos + dist_neg)
            
            results_df = pd.DataFrame({
                "Alternative": alt_names,
                "D+ (to ideal)": dist_pos,
                "D- (to anti-ideal)": dist_neg,
                "Score (Ci)": closeness
            })
            
            results_df = results_df.sort_values(by="Score (Ci)", ascending=False)
            results_df["Rank"] = range(1, len(results_df) + 1)
            
            st.success("Calculation done successfully!")
            
            st.subheader("3. Results and Ranking")
            st.dataframe(results_df.style.format({
                "D+ (to ideal)": "{:.4f}",
                "D- (to anti-ideal)": "{:.4f}",
                "Score (Ci)": "{:.4f}"
            }), use_container_width=True)
            
            st.bar_chart(results_df.set_index("Alternative")["Score (Ci)"])
            
            with st.expander("Search for interim calculations"):
                st.write("Normalized matrix")
                st.dataframe(pd.DataFrame(normalized_matrix, columns=crit_names, index=alt_names))
                st.write("Weighted normalized matrix")
                st.dataframe(pd.DataFrame(weighted_matrix, columns=crit_names, index=alt_names))
                st.write("Ideal solutions:")
                st.write(f"A+ (Best): {np.round(ideal_best, 4)}")
                st.write(f"A- (Worst): {np.round(ideal_worst, 4)}")

        except Exception as e:
            st.error(f"Error occured during calculation: {e}")

if __name__ == "__main__":
    run_topsis()