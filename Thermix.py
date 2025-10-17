import streamlit as st
import pandas as pd
import sympy as sy
from math import exp, isclose
import base64


st.markdown("""
<style>
h1 {
    font-size: 4rem !important;
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("Thermix")



engq_url = "https://i.imgur.com/k3fpXPI.png"
st.markdown(
    f"""
    <style>
    .corner-image {{
        position: absolute;
        top: -110px;
        right: 25px;
        width: 140px;
        height: 140px;
        border-radius: 8px;
        z-index: 999;
        border: 2px solid #fff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    </style>
    <img src="{engq_url}" class="corner-image">
    """,
    unsafe_allow_html=True
)


tab1, tab2 = st.tabs(["Página inicial", "📘 Manual de Utilização"])

with tab1:
    st.header("Quantos componentes?")
    n = st.number_input("Número de componentes (n)", min_value=2, max_value=10, step=1)

    st.header("Condições de operação:")
    col1, col2 = st.columns([1, 1]) 
    with col1:
        T = st.number_input("Temperatura (K)", min_value=0.01, format="%.2f", step=0.000001, value=273.15)
    with col2:
        P = st.number_input("Pressão (bar)", min_value=0.01, format="%.2f", step=0.000001, value=1.0)

    def n_float(x):
        try:
            return float(str(x).replace(',', '.'))
        except:
            return x

    if "parametros" not in st.session_state or st.session_state["parametros"].shape[1] != n:
        st.session_state["parametros"] = pd.DataFrame({col: [""] * n for col in ["Zi", "w", "Tc", "Pc", "Vc", "Zc", "A", "B", "C", "Vi"]}, index=[f"Componente {i+1}" for i in range(n)]).transpose()

    if "Aij" not in st.session_state or st.session_state["Aij"].shape != (n, n):
        st.session_state["Aij"] = pd.DataFrame([["" for _ in range(n)] for _ in range(n)],index=[f"{i+1}" for i in range(n)],columns=[f"{i+1}" for i in range(n)])

    st.header("Adicione os dados dos componentes:")
    edited_df_pmt = st.data_editor(st.session_state["parametros"], use_container_width=True)
    dfd = st.session_state["parametros"].applymap(n_float).transpose().to_dict('list')

    st.header("Adicione os coeficientes de interação:")
    edited_df_Aij = st.data_editor(st.session_state["Aij"], use_container_width=True)
    Aij = st.session_state["Aij"].applymap(n_float).to_numpy()

    if st.button("Salvar alterações"):
        st.session_state["parametros"] = edited_df_pmt.applymap(n_float)
        st.session_state["Aij"] = edited_df_Aij
        st.success("Parâmetros atualizados com sucesso!")


    st.write("\n")
    st.write("\n")

    ### Funções

    def wij(i, j):
        return (dfd['w'][i-1] + dfd['w'][j-1])/2


    def Zcij(i, j):
        return (dfd['Zc'][i-1] + dfd['Zc'][j-1])/2


    def Vcij(i, j):
        return sy.Pow(((sy.Pow(dfd['Vc'][i-1], 1/3) + sy.Pow(dfd['Vc'][j-1], 1/3))/2), 3) # type: ignore


    def kij(i, j):
        return 1 - sy.Pow((2*sy.Pow(dfd['Vc'][i-1]*dfd['Vc'][j-1], 1/6))/(sy.Pow(dfd['Vc'][i-1], 1/3) + sy.Pow(dfd['Vc'][j-1], 1/3)), 3) # type: ignore


    def Tcij(i, j):
        return (sy.Pow(dfd['Tc'][i-1]*dfd['Tc'][j-1], 0.5))*(1 - kij(i, j))


    def Pcij(i, j):
        return 83.14462*Tcij(i, j)*Zcij(i, j)/Vcij(i, j)


    def Pisat(T, i):
        return sy.Pow(10, dfd['A'][i-1] - (dfd['B'][i-1]/(T + dfd['C'][i-1])))


    def Tr(T, i, j):
        return T/Tcij(i, j)


    def Bij(T, i, j):
        return 83.14462*Tcij(i, j)/Pcij(i, j)*(0.083 - 0.422/Tr(T, i, j)**1.6 + wij(i, j)*(0.139 - 0.172/Tr(T, i, j)**4.2))


    def Cij(T, i, j):
        return (83.14462*Tcij(i, j)/Pcij(i, j))**2 * (0.01407 + 0.02432/Tr(T, i, j) - 0.00313/Tr(T, i, j)**10.5 + wij(i, j)*(-0.02676 + 0.05539/Tr(T, i, j)**2.7 - 0.00242/Tr(T, i, j)**10.5))


    def Cijk(T, i, j, k):
        return sy.real_root(Cij(T, i, j)*Cij(T, i, k)*Cij(T, j, k), 3)


    def Vsat(T, i):
        V = sy.symbols('V', real=True, positive=True)
        return sy.nsolve(sy.Eq(Pisat(T, i)*V/(83.14462*T), 1 + (Bij(T, i, i)/V) + (Cij(T, i, i)/(V**2))), V, 83.14462*T/Pisat(T, i))


    def Zsat(T, i):
        return Pisat(T, i)*Vsat(T, i)/(83.14462*T) # type: ignore


    def Bb(T, yi=(dfd['Zi']), n=len(dfd['w'])):
        return sum([yi[i]*yi[j]*Bij(T, i+1, j+1) for i in range(n) for j in range(n)])


    def Cc(T, yi=(dfd['Zi']), n=len(dfd['w'])):
        return sum([yi[i]*yi[j]*yi[k]*Cijk(T, i+1, j+1, k+1) for i in range(n) for j in range(n) for k in range(n)])


    def Vsat_mix(T, P0, n=len(dfd['w'])):
        V = sy.symbols('V', real=True, positive=True)
        return sy.nsolve(sy.Eq(P0 * V / (83.14462 * T), 1 + (Bb(T) / V) + (Cc(T) / (V ** 2))), V, 83.14462 * T / P0)


    def Zz(T, P, n=len(dfd['w'])):
        return P*Vsat_mix(T, P)/(83.14462*T)


    def Fii(P, T, i, yi=(dfd['Zi']), n=len(dfd['w'])):
        comeco = 2/Vsat_mix(T, P) * sum([yi[j]*Bij(T, i, j+1) for j in range(n)])
        meio = 3/(2*Vsat_mix(T, P)**2) * sum([yi[j]*yi[k]*Cijk(T, i, j+1, k+1) for j in range(n) for k in range(n)])
        return exp(comeco + meio - sy.ln(Zz(T, P)))


    def fiisat(T, i):
        return exp(2*Bij(T, i, i)/Vsat(T, i) + 1.5*Cij(T, i, i)/Vsat(T, i)**2 - sy.ln(Zsat(T, i)))


    def Omegaij(T, i, j):
        return dfd['Vi'][j-1]/dfd['Vi'][i-1] * exp(-Aij[i-1][j-1]/(8.314462*0.239006*T))


    def gamma(T, i, xi=dfd['Zi'], n=len(dfd['w'])):
        parte_1 = sy.ln(sum([xi[j]*Omegaij(T, i, j+1) for j in range(n)]))
        parte_2 = sum([xi[k]*Omegaij(T, k+1, i)/(sum([xi[j]*Omegaij(T, k+1, j+1) for j in range(n)])) for k in range(n)])
        return exp(sy.Integer(1) - parte_1 - parte_2)


    def PBOl(T, n=len(dfd['w'])):
        xi = dfd['Zi']
        P1 = 1000
        Coef_Fug = [1 for i in range(n)]
        Coef_Ativ = [gamma(T, i+1, xi) for i in range(n)]
        Pb = sum([Coef_Ativ[i]*xi[i]*Pisat(T, i+1)/Coef_Fug[i] for i in range(n)])
        while abs(P1 - Pb) > 0.001:
            P1 = Pb
            yi = [Coef_Ativ[i]*xi[i]*Pisat(T, i+1)/(Coef_Fug[i]*P1) for i in range(n)]
            Coef_Fug = [Fii(P1, T, i+1, yi)/fiisat(T, i+1) for i in range(n)]
            Pb = sum([Coef_Ativ[i] * xi[i] * Pisat(T, i+1) / Coef_Fug[i] for i in range(n)])

        return Pb, yi, Coef_Ativ, Coef_Fug


    def PORV(T, n=len(dfd['w'])):
        yi = dfd['Zi']
        P1 = 1000
        Coef_Fug = [1 for i in range(n)]
        Coef_Ativ0 = [1 for i in range(n)]
        Po = 1/sum([yi[i]*Coef_Fug[i]/(Coef_Ativ0[i]*Pisat(T,i+1)) for i in range(n)])
        xi = [yi[i]*Coef_Fug[i]*Po/(Coef_Ativ0[i]*Pisat(T, i+1)) for i in range(n)]
        Coef_Ativ = [gamma(T, i + 1, xi) for i in range(n)]
        while abs(Po - P1) > 0.001:
            P1 = Po
            Coef_Fug = [Fii(P1, T, i + 1, yi) / fiisat(T, i + 1) for i in range(n)]
            while all(abs(Coef_Ativ0[i] - Coef_Ativ[i]) > 0.001 for i in range(n)):
                Coef_Ativ0 = Coef_Ativ
                xi = [yi[i] * Coef_Fug[i] * Po / (Coef_Ativ0[i] * Pisat(T, i + 1)) for i in range(n)]
                xi = [xi[i]/sum(xi) for i in range(n)]
                Coef_Ativ = [gamma(T, i + 1, xi) for i in range(n)]
            xi = [yi[i] * Coef_Fug[i] * Po / (Coef_Ativ0[i] * Pisat(T, i + 1)) for i in range(n)]
            xi = [xi[i] / sum(xi) for i in range(n)]
            Coef_Ativ = [gamma(T, i + 1, xi) for i in range(n)]
            Po = 1 / sum([yi[i] * Coef_Fug[i] / (Coef_Ativ0[i] * Pisat(T, i + 1)) for i in range(n)])

        return Po, xi, Coef_Ativ, Coef_Fug


    def Flash(T, P, n=len(dfd['w'])):
        zi = dfd['Zi']
        Orv = list(PORV(T))
        Bol = list(PBOl(T))
        if P < Orv[0]:
            print(f'Pressão menor do que pressão de orvalho ({Orv[0]}), sistema totalmente gasoso')
        if P > Bol[0]:
            print(f'Pressão maior do que pressão de bolha ({Bol[0]}), sistema totalmente líquido')
        if Orv[0] <= P <= Bol[0]:
            V0 = (Bol[0] - P)/(Bol[0] - Orv[0])
            Coef_Ativ = [Orv[2][i] + (1-V0)*(Bol[2][i] - Orv[2][i]) for i in range(n)]
            Coef_Fug = [Orv[3][i] + (1-V0)*(Bol[3][i] - Orv[3][i]) for i in range(n)]
            Ki = [Coef_Ativ[i]*Pisat(T, i+1)/(Coef_Fug[i]*P) for i in range(n)]
            V = sy.symbols('V', real=True, positive=True)
            V = sy.nsolve(sum([zi[i] * (Ki[i] - 1) / (1 + V * (Ki[i] - 1)) for i in range(n)]), V, V0)
            while abs(V - V0) > 0.001:
                V0 = V
                xi = [zi[i]/(1 + V0*(Ki[i]-1)) for i in range(n)]
                yi = [Ki[i]*xi[i] for i in range(n)]
                Coef_Ativ = [gamma(T, i + 1, xi) for i in range(n)]
                Coef_Fug = [Fii(P, T, i + 1, yi) / fiisat(T, i + 1) for i in range(n)]
                Ki = [Coef_Ativ[i] * Pisat(T, i + 1) / (Coef_Fug[i] * P) for i in range(n)]
                V = sy.symbols('V', real=True, positive=True)
                V = sy.nsolve(sum([zi[i] * (Ki[i] - 1) / (1 + V * (Ki[i] - 1)) for i in range(n)]), V, V0)
            return V, xi, yi, Coef_Ativ, Coef_Fug, Ki, Orv[0], Bol[0]


    st.header("Escolha um calculo:")
    R_inter = st.selectbox("", options=[" ", "PBOL", "PORV", "FLASH"], label_visibility="collapsed")

    # Verificação de dados antes do cálculo
    def dados_validos(dfd, Aij, T, P):
        try:
            for col in dfd:
                if any(val == "" or not isinstance(val, (int, float)) for val in dfd[col]):
                    st.error(f"Coluna '{col}' contém valores inválidos. Preencha todos os campos com números.")
                    return False
                

            if any(any(val == "" or not isinstance(val, (int, float)) for val in linha) for linha in Aij):
                st.error("A matriz de interação Aij contém valores inválidos.")
                return False

            if not isclose(sum(dfd['Zi']), 1, rel_tol=1e-6):
                st.error("A soma das frações molares (Zi) deve ser igual a 1.")
                return False
            
            if T <= 0:
                st.error("A temperatura deve ser maior que zero.")
                return False
            
            if P <= 0:
                st.error("A pressão deve ser maior que zero.")
                return False
            
            return True
        except Exception as e:
            st.error(f"Ocorreu um erro na validação dos dados: {e}")
            return False


    disabled = any(val == "" for col in dfd for val in dfd[col]) or any("" in linha for linha in Aij)
    if st.button("Calcular", disabled=disabled):
        if not dados_validos(dfd, Aij, T, P):
            st.stop()
        else:
            pass
        with st.spinner("Calculando..."):
            if R_inter == "PBOL":
                resultado = PBOl(T)
                resultados_df = pd.DataFrame({
                    'Parâmetro': ['Pressão de Bolha'],
                    'Valor': [f"{resultado[0]:.4f} bar"]
                })
                composicao_df = pd.DataFrame({
                    'Componente': [f'Componente {i+1}' for i in range(len(resultado[1]))],
                    'xi': [f"{val:.6f}" for val in resultado[1]]
                })
                coeficientes_df = pd.DataFrame({
                    'Componente': [f'Componente {i+1}' for i in range(len(resultado[2]))],
                    'Coeficiente de atividade (γ)': [f"{val:.6f}" for val in resultado[2]],
                    'Coeficiente de fugacidade (Φ)': [f"{val:.6f}" for val in resultado[3]]
                })
                st.subheader("Resultados - Pressão de Bolha")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pressão de Bolha", f"{resultado[0]:.4f} bar")
                
                st.subheader("Composição Líquida")
                st.dataframe(composicao_df, use_container_width=True, hide_index=True)
                
                st.subheader("Coeficientes")
                st.dataframe(coeficientes_df, use_container_width=True, hide_index=True)

            elif R_inter == "PORV":
                resultado = PORV(T)
                composicao_df = pd.DataFrame({
                    'Componente': [f'Componente {i+1}' for i in range(len(resultado[1]))],
                    'yi': [f"{val:.6f}" for val in resultado[1]]})

                coeficientes_df = pd.DataFrame({
                    'Componente': [f'Componente {i+1}' for i in range(len(resultado[2]))],
                    'Coeficiente de atividade (γ)': [f"{val:.6f}" for val in resultado[2]],
                    'Coeficiente de fugacidade (Φ)': [f"{val:.6f}" for val in resultado[3]]})
 
                st.subheader("Resultados - Pressão de Orvalho")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pressão de Orvalho", f"{resultado[0]:.4f} bar")
                
                st.subheader("Composição Vapor")
                st.dataframe(composicao_df, use_container_width=True, hide_index=True)
                
                st.subheader("Coeficientes")
                st.dataframe(coeficientes_df, use_container_width=True, hide_index=True)


            elif R_inter == "FLASH":
                resultado = Flash(T, P)
                if resultado is not None:
                    composicoes_df = pd.DataFrame({
                        'Componente': [f'Componente {i+1}' for i in range(len(resultado[1]))],
                        'xi (Líquido)': [f"{val:.6f}" for val in resultado[1]],
                        'yi (Vapor)': [f"{val:.6f}" for val in resultado[2]],
                        'Ki': [f"{val:.6f}" for val in resultado[5]]})
                    
                    coeficientes_df = pd.DataFrame({
                        'Componente': [f'Componente {i+1}' for i in range(len(resultado[3]))],
                        'Coeficiente de Atividade (γ)': [f"{val:.6f}" for val in resultado[3]],
                        'Coeficiente de Fugacidade (Φ)': [f"{val:.6f}" for val in resultado[4]]})
                    
                    st.subheader("Resultados - Flash")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fração Vapor (V)", f"{resultado[0]:.4f}")
                    with col2:
                        st.metric("Pressão Bolha", f"{resultado[7]:.4f} bar")
                    with col3:
                        st.metric("Pressão Orvalho", f"{resultado[6]:.4f} bar")
                    
                    st.subheader("Composições e Constantes de Equilíbrio")
                    st.dataframe(composicoes_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Coeficientes")
                    st.dataframe(coeficientes_df, use_container_width=True, hide_index=True)
                else:
                    Pbol, Porv = PBOl(T)[0], PORV(T)[0]
                    st.warning("A pressão está fora do intervalo válido para o cálculo do flash.")
                    if P < Porv:
                        st.warning("Pressão de operação menor do que a pressão de orvalho, sistema totalmente gasoso.")
                    elif P > Pbol:
                        st.warning("Pressão de operação maior do que a pressão de bolha, sistema totalmente líquido.")
                    st.warning(f"Pressão de orvalho: {round(Porv, 2)} bar, Pressão de bolha: {round(Pbol, 2)} bar.")


with tab2:
    st.header("Manual de Utilização")
    pdf_url = "https://drive.google.com/uc?export=download&id=1o445Akz2k9wEfSX4eqwB6kJB175HUkg3"

    # Botão de download
    st.markdown(
        f"""
        <a href="{pdf_url}" target="_blank">
            <button style="
                background-color:#4CAF50;
                color:white;
                border:none;
                padding:12px 24px;
                text-align:center;
                text-decoration:none;
                display:inline-block;
                font-size:16px;
                border-radius:8px;
                cursor:pointer;
            ">
            🡇­­­­­­ ­ Baixar Manual de Utilização
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

