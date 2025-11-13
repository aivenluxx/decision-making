import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Реалізація Fuzzy TOPSIS (з CSV)", layout="wide")

SCALE_VALUES = {
    "Level 1": (1, 1, 3),
    "Level 2": (1, 3, 5),
    "Level 3": (3, 5, 7),
    "Level 4": (5, 7, 9),
    "Level 5": (7, 9, 9)
}

CRITERIA_SCALE = {
    "VL (Very Low)": "Level 1",
    "L (Low)": "Level 2",
    "M (Medium)": "Level 3",
    "H (High)": "Level 4",
    "VH (Very High)": "Level 5"
}

ALT_SCALE = {
    "VP (Very Poor)": "Level 1",
    "P (Poor)": "Level 2",
    "F (Fair)": "Level 3",
    "G (Good)": "Level 4",
    "VG (Very Good)": "Level 5"
}

def get_fuzzy_val(label, mapping):
    level = mapping.get(label)
    if not level:
        for k, v in mapping.items():
            if k.startswith(label):
                level = v
                break
    return SCALE_VALUES.get(level, (1, 1, 1))

def fuzzy_add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2])

def fuzzy_div_scalar(t1, k):
    return (t1[0] / k, t1[1] / k, t1[2] / k)

def fuzzy_mul(t1, t2):
    return (t1[0] * t2[0], t1[1] * t2[1], t1[2] * t2[2])

def distance_vertex(t1, t2):
    return math.sqrt((1/3) * ((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2))

st.title("Реалізація Fuzzy TOPSIS")
st.markdown("""
Цей застосунок імплементує метод **Fuzzy TOPSIS** для багатокритеріального прийняття рішень.
Ви можете ввести дані вручну або завантажити CSV файли.
""")

with st.sidebar:
    st.header("⚙️ Конфігурація")
    st.info("Сконфігуруйте параметри, як необхідно (за замовчуванням: 5 критеріїв, 4 альтернативи , 3 експерти).")
    
    num_experts = st.number_input("Кількість експертів", min_value=3, value=3)
    num_criteria = st.number_input("Кількість критеріїв", min_value=5, value=5)
    num_alternatives = st.number_input("Кількість альтернатив", min_value=4, value=4)

    st.markdown("---")
    crit_names = [st.text_input(f"Критерій {i+1} ", value=f"C{i+1}") for i in range(num_criteria)]
    alt_names = [st.text_input(f"Альтернатива {i+1} ", value=f"A{i+1}") for i in range(num_alternatives)]

st.header("1. Введення даних")

st.subheader("A. Ваги критеріїв")
st.markdown("Вкажіть, наскільки важливим є цей критерій (від VL до VH).")

uploaded_weights = st.file_uploader("Завантажити ваги (CSV)", type=["csv"], help="Формат: Рядки - Експерти, Стовпці - Критерії. Значення: VL, L, M, H, VH")

if uploaded_weights is not None:
    try:
        df_weights_input = pd.read_csv(uploaded_weights)
        if df_weights_input.shape[1] > num_criteria: 
             df_weights_input = df_weights_input.iloc[:, 1:]
        
        if df_weights_input.shape != (num_experts, num_criteria):
             st.warning(f"Розмір завантаженого файлу ({df_weights_input.shape}) не співпадає з налаштуваннями ({num_experts}x{num_criteria}). Буде використано лише частину даних або додано пусті.")
             df_weights_input = df_weights_input.iloc[:num_experts, :num_criteria]
        
        df_weights_input.columns = crit_names
        df_weights_input.index = [f"Експерт {i+1}" for i in range(num_experts)]
        st.success("Ваги успішно завантажено!")
    except Exception as e:
        st.error(f"Помилка читання CSV ваг: {e}")
        df_weights_input = pd.DataFrame(index=[f"Експерт {i+1}" for i in range(num_experts)], columns=crit_names)
else:
    df_weights_input = pd.DataFrame(
        index=[f"Експерт {i+1}" for i in range(num_experts)],
        columns=crit_names
    )

weight_config = {col: st.column_config.SelectboxColumn(col, options=list(CRITERIA_SCALE.keys()), required=True) for col in crit_names}
edited_weights = st.data_editor(df_weights_input, column_config=weight_config, use_container_width=True, key="weights_editor")


st.subheader("B. Оцінка альтернатив")
st.markdown("Оцініть у порівнянні одну альтернативу до іншої (від VP до VG).")


uploaded_ratings = st.file_uploader("Завантажити оцінки (CSV)", type=["csv"], help="Формат колонок: 'Expert', 'Alternative', 'C1', 'C2'...")

expert_inputs_data = {}

if uploaded_ratings is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_ratings)
        if 'Expert' not in df_uploaded.columns or 'Alternative' not in df_uploaded.columns:
            st.error("CSV файл оцінок повинен містити колонки 'Expert' та 'Alternative'.")
        else:
            cols_data = df_uploaded.columns[2:]
            if len(cols_data) != num_criteria:
                 st.warning("Кількість критеріїв у файлі не співпадає з налаштуваннями.")
            
            for i in range(num_experts):
                expert_label = f"Експерт {i+1}"
                
                start_row = i * num_alternatives
                end_row = start_row + num_alternatives
                
                if end_row <= len(df_uploaded):
                    subset = df_uploaded.iloc[start_row:end_row].copy()
                    subset.index = alt_names 
                    subset_vals = subset.iloc[:, 2:2+num_criteria]
                    subset_vals.columns = crit_names
                    expert_inputs_data[i] = subset_vals
                else:
                    expert_inputs_data[i] = None
            st.success("Оцінки успішно завантажено!")

    except Exception as e:
        st.error(f"Помилка читання CSV оцінок: {e}")

expert_inputs = {}
tabs = st.tabs([f"Експерт {i+1}" for i in range(num_experts)])

for i, tab in enumerate(tabs):
    with tab:
        st.write(f"**Рейтинг експерта {i+1}:**")
        
        if i in expert_inputs_data and expert_inputs_data[i] is not None:
            df_alt_input = expert_inputs_data[i]
        else:
            df_alt_input = pd.DataFrame(
                index=alt_names,
                columns=crit_names
            )
            
        alt_config = {col: st.column_config.SelectboxColumn(col, options=list(ALT_SCALE.keys()), required=True) for col in crit_names}
        
        expert_inputs[i] = st.data_editor(df_alt_input, column_config=alt_config, use_container_width=True, key=f"exp_{i}_editor")



if st.button("🚀 Розрахувати оцінку Fuzzy TOPSIS"):
    
    if edited_weights.isnull().values.any():
        st.error("Будь ласка, заповніть всі ваги критеріїв.")
        st.stop()

    for i in range(num_experts):
        if expert_inputs[i].isnull().values.any():
            st.error(f"Будь ласка, заповніть оцінки для Експерта {i+1}.")
            st.stop()

    try:

        agg_weights = {} 
        
        for col in crit_names:
            sum_f = (0, 0, 0)
            for idx, row in edited_weights.iterrows():
                val = get_fuzzy_val(row[col], CRITERIA_SCALE)
                sum_f = fuzzy_add(sum_f, val)
            agg_weights[col] = fuzzy_div_scalar(sum_f, num_experts)
            
        agg_matrix = {}
        
        for alt in alt_names:
            for crit in crit_names:
                sum_f = (0, 0, 0)
                for i in range(num_experts):
                    val_str = expert_inputs[i].loc[alt, crit]
                    val_f = get_fuzzy_val(val_str, ALT_SCALE)
                    sum_f = fuzzy_add(sum_f, val_f)
                agg_matrix[(alt, crit)] = fuzzy_div_scalar(sum_f, num_experts)
       
        norm_matrix = {}
        
        for crit in crit_names:
            max_c = 0
            for alt in alt_names:
                val = agg_matrix[(alt, crit)]
                if val[2] > max_c: 
                    max_c = val[2]

            for alt in alt_names:
                val = agg_matrix[(alt, crit)]
                if max_c == 0: max_c = 1 
                norm_matrix[(alt, crit)] = (val[0]/max_c, val[1]/max_c, val[2]/max_c)
        
        weighted_matrix = {}
        
        for alt in alt_names:
            for crit in crit_names:
                r = norm_matrix[(alt, crit)]
                w = agg_weights[crit]
                weighted_matrix[(alt, crit)] = fuzzy_mul(r, w)

        fpis = {} 
        fnis = {} 
        
        for crit in crit_names:
            col_vals = [weighted_matrix[(alt, crit)] for alt in alt_names]
            max_val = max([v[2] for v in col_vals])
            min_val = min([v[0] for v in col_vals])
            
            fpis[crit] = (max_val, max_val, max_val)
            fnis[crit] = (min_val, min_val, min_val)
        
        results = []
        
        for alt in alt_names:
            d_star = 0 
            d_minus = 0 
            
            for crit in crit_names:
                v_ij = weighted_matrix[(alt, crit)]
                
                dist_to_fpis = distance_vertex(v_ij, fpis[crit])
                dist_to_fnis = distance_vertex(v_ij, fnis[crit])
                
                d_star += dist_to_fpis
                d_minus += dist_to_fnis
            
            if (d_star + d_minus) == 0:
                cc = 0
            else:
                cc = d_minus / (d_star + d_minus)
            
            results.append({
                "Альтернатива": alt,
                "D* (Відстань до найкращого)": round(d_star, 4),
                "D- (Відстань до найгіршого)": round(d_minus, 4),
                "Коєфіцієнт близькості": round(cc, 4)
            })
      
        st.markdown("---")
        st.header("Результати")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Коєфіцієнт близькості", ascending=False).reset_index(drop=True)
        
        winner = df_results.iloc[0]['Альтернатива']
        st.success(f"Найкраща альтернатива: **{winner}**")
        
        st.dataframe(df_results, use_container_width=True)

        with st.expander("Показати детальні розрахункові матриці"):
            st.write("**Агреговані ваги (Fuzzy):**", agg_weights)
            
            st.write(f"**Взважені нормалізовані матриці (Приклад - {alt_names[0]}):**")
            first_alt = alt_names[0]
            
            display_matrix = {str(k[1]): v for k, v in weighted_matrix.items() if k[0] == first_alt}
            
            st.write(display_matrix)

    except Exception as e:
        st.error(f"Під час розрахунків сталася помилка: {e}")

else:
    st.info("Заповніть таблиці або завантажте CSV файли і натисніть кнопку 'Розрахувати'.")