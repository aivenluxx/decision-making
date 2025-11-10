import streamlit as st
import pandas as pd
import numpy as np

LINGUISTIC_SCALE_WEIGHTS = {
    "VL": (0.0, 0.1, 0.3),
    "L": (0.1, 0.3, 0.5),
    "M": (0.3, 0.5, 0.7),
    "H": (0.5, 0.7, 0.9),
    "VH": (0.7, 0.9, 1.0)
}

LINGUISTIC_SCALE_ALTERNATIVES = {
    "VP": (0.0, 0.0, 0.2),
    "P": (0.0, 0.2, 0.4),
    "F": (0.2, 0.4, 0.6),
    "G": (0.4, 0.6, 0.8),
    "VG": (0.6, 0.8, 1.0),
    "E": (0.8, 0.9, 1.0)
}

def aggregate_fuzzy_numbers(fuzzy_numbers):
   
    if not fuzzy_numbers:
        return (0, 0, 0)
    
    l_values = [f[0] for f in fuzzy_numbers]
    m_values = [f[1] for f in fuzzy_numbers]
    r_values = [f[2] for f in fuzzy_numbers]
    
    p = len(fuzzy_numbers)
    
    l_agg = min(l_values)
    m_agg = sum(m_values) / p
    r_agg = max(r_values)
    
    return (l_agg, m_agg, r_agg)

def fuzzy_subtract(f1, f2):
    
    return (f1[0] - f2[2], f1[1] - f2[1], f1[2] - f2[0])

def fuzzy_divide_scalar(f, scalar):
    
    if scalar == 0:
        return (0, 0, 0)
    return (f[0] / scalar, f[1] / scalar, f[2] / scalar)

def fuzzy_multiply_fuzzy(f1, f2):
    
    return (f1[0] * f2[0], f1[1] * f2[1], f1[2] * f2[2])

def fuzzy_add(f1, f2):
    
    return (f1[0] + f2[0], f1[1] + f2[1], f1[2] + f2[2])

def fuzzy_multiply_scalar(f, scalar):
    
    return (f[0] * scalar, f[1] * scalar, f[2] * scalar)

def defuzzify(f):
   
    return (f[0] + 2 * f[1] + f[2]) / 4


st.set_page_config(layout="wide", page_title="Fuzzy VIKOR")
st.title("Реалізація методу VIKOR")
st.write("Цей застосунок реалізує метод групового експертного оцінювання VIKOR.")

st.sidebar.header("Параметри задачі")

n_alternatives = st.sidebar.number_input("Кількість альтернатив (m)", min_value=2, value=8)
n_criteria = st.sidebar.number_input("Кількість критеріїв (n)", min_value=2, value=6)
n_experts = st.sidebar.number_input("Кількість експертів (p)", min_value=1, value=4)

v = st.sidebar.slider("Вага компромісної стратегії (ν)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                    help="ν > 0.5 - перевага 'максимальній груповій корисності' (S), ν < 0.5 - перевага 'мінімальним індивідуальним втратам' (R).")

alternative_names = [f"A{i+1}" for i in range(n_alternatives)]
criteria_names = [f"C{i+1}" for i in range(n_criteria)]
expert_names = [f"D{i+1}" for i in range(n_experts)]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Крок 1: Введення даних",
    "Крок 2: Агреговані матриці",
    "Крок 3-4: Ідеальні та нормовані значення",
    "Крок 5-7: Обчислення S, R, Q",
    "Крок 8-9: Ранжування та результати"
])

with tab1:
    st.header("Крок 1: Визначення лінгвістичних змінних та збір оцінок")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Лінгвістична шкала важливості критеріїв")
       
        st.dataframe(pd.DataFrame(LINGUISTIC_SCALE_WEIGHTS.values(), 
                                    index=LINGUISTIC_SCALE_WEIGHTS.keys(), 
                                    columns=["l", "m", "r"]), use_container_width=True)
    with col2:
        st.subheader("Лінгвістична шкала оцінок альтернатив")
        
        st.dataframe(pd.DataFrame(LINGUISTIC_SCALE_ALTERNATIVES.values(), 
                                    index=LINGUISTIC_SCALE_ALTERNATIVES.keys(), 
                                    columns=["l", "m", "r"]), use_container_width=True)
    
    st.info("Будь ласка, заповніть наступні таблиці оцінок. Використовуйте скорочення, наприклад 'VL', 'H', 'VP', 'G' тощо. \n\n"
            "**Ви можете завантажити дані з CSV-файлів.** Файли повинні мати індексний стовпець з назвами критеріїв.")
           
    st.subheader("Визначення типу критеріїв")
    benefit_cost_df = pd.DataFrame(
        {"Тип": ["Benefit"] * n_criteria},
        index=criteria_names
    )
    edited_benefit_cost_df = st.data_editor(
        benefit_cost_df,
        column_config={
            "Тип": st.column_config.SelectboxColumn(
                "Тип критерію",
                options=["Benefit", "Cost"],
                required=True
            )
        },
        use_container_width=True
    )
    benefit_cost_map = edited_benefit_cost_df["Тип"].to_list()

   
    st.subheader("Оцінки важливості критеріїв експертами (W)")
    
    
    weights_df_default = pd.DataFrame(
        {expert: ["M"] * n_criteria for expert in expert_names},
        index=criteria_names
    )
        
    weights_uploader = st.file_uploader(
        f"Завантажити CSV з вагами (очікувана форма: {n_criteria} рядків, {n_experts} стовпців)", 
        type=["csv"], 
        key="weights_csv"
    )
    
    weights_df_to_edit = weights_df_default 
    
    if weights_uploader is not None:
        try:
            df_from_csv = pd.read_csv(weights_uploader, index_col=0).applymap(lambda x: x.strip() if isinstance(x, str) else x) # Додано .strip()
                        
            if (df_from_csv.shape == (n_criteria, n_experts) and 
                list(df_from_csv.index) == criteria_names and 
                list(df_from_csv.columns) == expert_names):
                weights_df_to_edit = df_from_csv
                st.success("Файл з вагами успішно завантажено.")
            else:
                st.warning(f"Структура CSV файлу не відповідає параметрам задачі. "
                           f"Очікувалось: {n_criteria} рядків (індекс: {', '.join(criteria_names)}) та "
                           f"{n_experts} стовпців ({', '.join(expert_names)}). "
                           f"Завантажено: {df_from_csv.shape[0]}x{df_from_csv.shape[1]}. "
                           "Використовуються значення за замовчуванням.")
        except Exception as e:
            st.error(f"Помилка читання файлу ваг: {e}. Переконайтеся, що перший стовпець є індексним.")
    
    edited_weights_df = st.data_editor(
        weights_df_to_edit,
        column_config={
            expert: st.column_config.SelectboxColumn(
                f"Оцінка {expert}",
                options=list(LINGUISTIC_SCALE_WEIGHTS.keys()),
                required=True
            ) for expert in expert_names
        },
        use_container_width=True
    )
    
    st.subheader("Оцінки альтернатив по критеріях експертами (D)")
    
    expert_tabs = st.tabs(expert_names)
    expert_ratings_dfs = []
    
    for i, expert_tab in enumerate(expert_tabs):
        with expert_tab:
            st.write(f"Будь ласка, заповніть матрицю оцінок для експерта {expert_names[i]}")
                        
            default_ratings_df = pd.DataFrame(
                {alt: ["F"] * n_criteria for alt in alternative_names},
                index=criteria_names
            )
                        
            ratings_uploader = st.file_uploader(
                f"Завантажити CSV для {expert_names[i]} (очікувана форма: {n_criteria} рядків, {n_alternatives} стовпців)", 
                type=["csv"], 
                key=f"expert_{i}_ratings_csv"
            )
            
            ratings_df_to_edit = default_ratings_df 
            
            if ratings_uploader is not None:
                try:
                    df_from_csv = pd.read_csv(ratings_uploader, index_col=0).applymap(lambda x: x.strip() if isinstance(x, str) else x) # Додано .strip()
                    
                    if (df_from_csv.shape == (n_criteria, n_alternatives) and 
                        list(df_from_csv.index) == criteria_names and 
                        list(df_from_csv.columns) == alternative_names):
                        ratings_df_to_edit = df_from_csv
                        st.success(f"Файл для {expert_names[i]} успішно завантажено.")
                    else:
                        st.warning(f"Структура CSV файлу не відповідає параметрам задачі. "
                                   f"Очікувалось: {n_criteria} рядків (індекс: {', '.join(criteria_names)}) та "
                                   f"{n_alternatives} стовпців ({', '.join(alternative_names)}). "
                                   f"Завантажено: {df_from_csv.shape[0]}x{df_from_csv.shape[1]}. "
                                   "Використовуються значення за замовчуванням.")
                except Exception as e:
                    st.error(f"Помилка читання файлу оцінок: {e}. Переконайтеся, що перший стовпець є індексним.")
            
            edited_ratings_df = st.data_editor(
                ratings_df_to_edit,
                key=f"expert_{i}_ratings",
                column_config={
                    alt: st.column_config.SelectboxColumn(
                        alt, 
                        options=list(LINGUISTIC_SCALE_ALTERNATIVES.keys()),
                        required=True
                    ) for alt in alternative_names
                },
                use_container_width=True
            )
            expert_ratings_dfs.append(edited_ratings_df)
    
    if st.button("РОЗРАХУВАТИ FUZZY VIKOR", type="primary", use_container_width=True):
        
        try:
            
            agg_weights = []
            for i in range(n_criteria):
                expert_linguistic_weights = edited_weights_df.iloc[i].values
                expert_fuzzy_weights = [LINGUISTIC_SCALE_WEIGHTS[w.strip()] for w in expert_linguistic_weights] 
                agg_weights.append(aggregate_fuzzy_numbers(expert_fuzzy_weights))
            
            df_agg_weights = pd.DataFrame(
                agg_weights, 
                index=criteria_names, 
                columns=["l", "m", "r"]
            )
                       
            agg_ratings = [[] for _ in range(n_alternatives)]
            for i in range(n_criteria):
                for j in range(n_alternatives):
                    expert_linguistic_ratings = [df.iloc[i, j] for df in expert_ratings_dfs]
                    expert_fuzzy_ratings = [LINGUISTIC_SCALE_ALTERNATIVES[r.strip()] for r in expert_linguistic_ratings] 
                    agg_ratings[j].append(aggregate_fuzzy_numbers(expert_fuzzy_ratings))

            df_agg_ratings = pd.DataFrame(
                agg_ratings,
                index=alternative_names,
                columns=criteria_names
            )

            with tab2:
                st.header("Крок 2: Агреговані нечіткі матриці")
                st.subheader("Агрегований нечіткий вектор ваги (W)")
                st.dataframe(df_agg_weights, use_container_width=True)
                
                st.subheader("Агрегована нечітка матриця продуктивності (D)")
                st.dataframe(df_agg_ratings, use_container_width=True)

                       
            f_star_list = []
            f_circ_list = []
            
            for i in range(n_criteria):
                criterion_col = df_agg_ratings.iloc[:, i]
                l_values = [f[0] for f in criterion_col]
                m_values = [f[1] for f in criterion_col]
                r_values = [f[2] for f in criterion_col]
                
                if benefit_cost_map[i] == "Benefit":
                    f_star = (max(l_values), max(m_values), max(r_values))
                    f_circ = (min(l_values), min(m_values), min(r_values))
                else: 
                    f_star = (min(l_values), min(m_values), min(r_values))
                    f_circ = (max(l_values), max(m_values), max(r_values))
                
                f_star_list.append(f_star)
                f_circ_list.append(f_circ)

            df_ideal = pd.DataFrame(
                {"f* (Ідеальне)": f_star_list, "f° (Найгірше)": f_circ_list},
                index=criteria_names
            )
                       
            norm_diff = [[] for _ in range(n_alternatives)]
            
            for j in range(n_alternatives):
                for i in range(n_criteria):
                    f_ij = agg_ratings[j][i]
                    f_star = f_star_list[i]
                    f_circ = f_circ_list[i]
                    
                    if benefit_cost_map[i] == "Benefit":
                    
                        numerator = fuzzy_subtract(f_star, f_ij)
                        denominator = f_star[2] - f_circ[0]
                    else: 
                        
                        numerator = fuzzy_subtract(f_ij, f_star)
                        denominator = f_circ[2] - f_star[0]
                                        
                    if denominator == 0:
                        d_ij = (0, 0, 0)
                    else:
                        d_ij = fuzzy_divide_scalar(numerator, denominator)
                    norm_diff[j].append(d_ij)
            
            df_norm_diff = pd.DataFrame(
                norm_diff,
                index=alternative_names,
                columns=criteria_names
            )

            with tab3:
                st.header("Крок 3: Ідеальні та найгірші значення")
                st.dataframe(df_ideal, use_container_width=True)
                
                st.header("Крок 4: Нормована нечітка різниця (d)")
                st.dataframe(df_norm_diff, use_container_width=True)
                        
            S_list = []
            R_list = []
            
            for j in range(n_alternatives):
                S_j = (0, 0, 0)
                R_j_components = []
                
                for i in range(n_criteria):
                    w_i = agg_weights[i]
                    d_ij = norm_diff[j][i]
                  
                 
                    weighted_d = fuzzy_multiply_fuzzy(w_i, d_ij)
                    
                    
                    S_j = fuzzy_add(S_j, weighted_d)
                    
                    
                    R_j_components.append(weighted_d)
                
                S_list.append(S_j)
                                
                R_l = max(f[0] for f in R_j_components)
                R_m = max(f[1] for f in R_j_components)
                R_r = max(f[2] for f in R_j_components)
                R_list.append((R_l, R_m, R_r))

                    
            
            S_star_l = min(f[0] for f in S_list)
            S_star_m = min(f[1] for f in S_list)
            S_star_r = min(f[2] for f in S_list)
            S_star_fuzzy = (S_star_l, S_star_m, S_star_r)
            S_circ_r = max(f[2] for f in S_list)
            
            
            R_star_l = min(f[0] for f in R_list)
            R_star_m = min(f[1] for f in R_list)
            R_star_r = min(f[2] for f in R_list)
            R_star_fuzzy = (R_star_l, R_star_m, R_star_r)
            R_circ_r = max(f[2] for f in R_list)

            Q_list = []
            
            den_S = S_circ_r - S_star_l
            den_R = R_circ_r - R_star_l

            for j in range(n_alternatives):
                
                num_S = fuzzy_subtract(S_list[j], S_star_fuzzy)
                if den_S == 0:
                    term_S_fuzzy = (0,0,0)
                else:
                    term_S_fuzzy = fuzzy_divide_scalar(num_S, den_S)
                term_S_weighted = fuzzy_multiply_scalar(term_S_fuzzy, v)
                
                
                num_R = fuzzy_subtract(R_list[j], R_star_fuzzy)
                if den_R == 0:
                    term_R_fuzzy = (0,0,0)
                else:
                    term_R_fuzzy = fuzzy_divide_scalar(num_R, den_R)
                term_R_weighted = fuzzy_multiply_scalar(term_R_fuzzy, (1 - v))
                                
                Q_j = fuzzy_add(term_S_weighted, term_R_weighted)
                Q_list.append(Q_j)
            
            df_fuzzy_srq = pd.DataFrame(
                {"S": S_list, "R": R_list, "Q": Q_list},
                index=alternative_names
            )
            
            S_crisp = [defuzzify(f) for f in S_list]
            R_crisp = [defuzzify(f) for f in R_list]
            Q_crisp = [defuzzify(f) for f in Q_list]
            
            df_crisp_srq = pd.DataFrame(
                {"S": S_crisp, "R": R_crisp, "Q": Q_crisp},
                index=alternative_names
            )

            with tab4:
                st.header("Крок 5-6: Нечіткі значення S, R та Q")
                st.dataframe(df_fuzzy_srq, use_container_width=True)
                
                st.header("Крок 7: Дефазифіковані (чіткі) значення S, R та Q")
                st.dataframe(df_crisp_srq, use_container_width=True)

                       
            df_ranks = pd.DataFrame(index=alternative_names)
            df_ranks["S"] = pd.Series(S_crisp, index=alternative_names).rank(method='min').astype(int)
            df_ranks["R"] = pd.Series(R_crisp, index=alternative_names).rank(method='min').astype(int)
            df_ranks["Q"] = pd.Series(Q_crisp, index=alternative_names).rank(method='min').astype(int)

            with tab5:
                st.header("Крок 8: Ранжування за S, R та Q")
                st.dataframe(df_ranks, use_container_width=True)
                               
                st.header("Крок 9: Визначення компромісного рішення")
                                
                results_df = pd.DataFrame({
                    "Q": Q_crisp,
                    "S_rank": df_ranks["S"],
                    "R_rank": df_ranks["R"]
                }, index=alternative_names).sort_values(by="Q")
                
                A1_name = results_df.index[0]
                A2_name = results_df.index[1]
                
                Q_A1 = results_df.iloc[0]["Q"]
                Q_A2 = results_df.iloc[1]["Q"]
                                
                Adv = Q_A2 - Q_A1
                DQ = 1 / (n_alternatives - 1)
                condition_1 = (Adv >= DQ)
                
                st.write(f"**Найкраща альтернатива (A(1)):** `{A1_name}` (Q = {Q_A1:.4f})")
                st.write(f"**Друга альтернатива (A(2)):** `{A2_name}` (Q = {Q_A2:.4f})")
                st.write("---")
                
                st.subheader("Перевірка умов")
                st.write(f"**Умова 1: 'Прийнятна перевага'**")
                st.markdown(f"* `Adv = Q(A(2)) - Q(A(1)) = {Q_A2:.4f} - {Q_A1:.4f} = {Adv:.4f}`")
                st.markdown(f"* `DQ = 1 / (m - 1) = 1 / ({n_alternatives} - 1) = {DQ:.4f}`")
                st.markdown(f"* **Умова 1 виконана:** `{condition_1}` (Adv ≥ DQ)")
                                
                A1_S_rank = results_df.loc[A1_name]["S_rank"]
                A1_R_rank = results_df.loc[A1_name]["R_rank"]
                condition_2 = (A1_S_rank == 1) or (A1_R_rank == 1)
                
                st.write(f"**Умова 2: 'Прийнятна стабільність'**")
                st.markdown(f"* Альтернатива `{A1_name}` (найкраща за Q) також повинна бути найкращою за S або R.")
                st.markdown(f"* Ранг `{A1_name}` за S: **{A1_S_rank}**")
                st.markdown(f"* Ранг `{A1_name}` за R: **{A1_R_rank}**")
                st.markdown(f"* **Умова 2 виконана:** `{condition_2}`")
                st.write("---")
               
                st.subheader("Компромісне рішення")
                if condition_1 and condition_2:
                    st.success(f"**Найкраще компромісне рішення: {A1_name}**")
                    st.write("Обидві умови виконані.")
                
                elif not condition_1:
                    st.error(f"**Умова 1 не виконана.** Пропонується набір компромісних рішень.")
                    solution_list = [A1_name]
                    for k in range(1, n_alternatives):
                        A_k_name = results_df.index[k]
                        Q_A_k = results_df.iloc[k]["Q"]
                        if (Q_A_k - Q_A1) < DQ:
                            if A_k_name not in solution_list:
                                solution_list.append(A_k_name)
                        else:
                            break
                    st.write(f"**Набір рішень (де Q(A(m)) - Q(A(1)) < DQ):** `{', '.join(solution_list)}`")
                
                elif not condition_2:
                    st.warning(f"**Умова 2 не виконана.** Пропонується набір компромісних рішень.")
                    st.write(f"**Набір рішень:** `{A1_name}, {A2_name}`")

        except KeyError as e:
            st.error(f"Під час розрахунків сталася помилка 'KeyError': Не вдалося знайти ключ {e}.")
            st.warning(f"Це означає, що у ваших CSV-файлах або в таблицях вище є значення, якого немає у словниках (наприклад, 'vh' замість 'VH', або 'good' замість 'G', або 'VL' у таблиці альтернатив). Будь ласка, перевірте дані.")
        except Exception as e:
            st.error(f"Під час розрахунків сталася неочікувана помилка: {e}")
            st.warning("Будь ласка, перевірте, чи всі лінгвістичні оцінки введено коректно.")