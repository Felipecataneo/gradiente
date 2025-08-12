import streamlit as st
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

# --- Configurações da Página e Funções Matemáticas ---

st.set_page_config(layout="wide", page_title="Descida do Gradiente Interativa")

# A nossa "montanha"
def f(x, y):
    return x * y * np.exp(-x**2 - y**2)

# A nossa "bússola" que aponta para a subida mais íngreme
def grad_f(x, y):
    exp_term = np.exp(-x**2 - y**2)
    df_dx = y * exp_term * (1 - 2*x**2)
    df_dy = x * exp_term * (1 - 2*y**2)
    return np.array([df_dx, df_dy])

# --- Interface do Usuário (Sidebar) ---

st.sidebar.title("Painel de Controle 🕹️")
st.sidebar.markdown("Use os controles abaixo para ajustar os parâmetros do algoritmo.")

# Controles do usuário na barra lateral
st.sidebar.header("Ponto Inicial (x₀)")
x0_val = st.sidebar.slider("Valor inicial de x", -2.0, 2.0, 0.3, 0.1)
y0_val = st.sidebar.slider("Valor inicial de y", -2.0, 2.0, 1.2, 0.1)
ponto_inicial = np.array([x0_val, y0_val])

st.sidebar.header("Parâmetros do Algoritmo")
lambda_passo = st.sidebar.slider("Tamanho do Passo (λ)", 0.01, 1.0, 0.1, 0.01, help="Define o 'tamanho' de cada passo que damos para descer a montanha. Passos grandes podem ser mais rápidos, mas podem errar o alvo. Passos pequenos são mais seguros, mas demoram mais.")
num_iteracoes = st.sidebar.slider("Número de Iterações (Passos)", 1, 100, 2, 1, help="Quantos passos queremos dar na nossa descida. Aumente para ver o caminho completo até o fundo do vale!")

# --- Corpo Principal do Aplicativo ---

st.title("🏔️ Aprendendo Gradiente Descendente de Forma Visual")
st.markdown("Bem-vindo! Vamos explorar como um computador 'aprende' a encontrar o ponto mais baixo de um terreno, usando uma técnica chamada **Descida do Gradiente** (ou Gradiente Descendente).")
st.info("**A Analogia:** Imagine que você está em uma montanha com neblina e quer chegar ao vale (o ponto mais baixo). Você só consegue ver o chão aos seus pés. A melhor estratégia é sentir a inclinação e dar um passo na direção que desce mais rápido. Repetindo isso, você eventualmente chegará ao fundo. É exatamente isso que faremos aqui!")

# --- Passo 1: A Função (Nossa Montanha) ---
st.header("Passo 1: Conhecendo o Terreno (A Função)")
st.markdown("O nosso 'terreno' ou 'montanha' é definido por uma função matemática. Queremos encontrar o par `(x, y)` que resulta no menor valor `f(x, y)`.")
st.latex(r"f(x, y) = x y e^{-x^2 - y^2}")

# Plot da função
x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

fig = matplotlib.pyplot.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), color='red', s=150, ec='black', label=f'Ponto Inicial ({x0_val:.2f}, {y0_val:.2f})', zorder=5)
ax.set_title("Visualização 3D da nossa 'Montanha'")
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Altitude f(x,y)")
ax.legend()
st.pyplot(fig)


# --- Passo 2: A Bússola (O Gradiente e as Derivadas) ---
st.header("Passo 2: A Bússola Mágica (O Gradiente)")
st.markdown("Para saber para onde descer, precisamos de uma 'bússola'. Essa bússola é o **Gradiente (∇f)**. Ele sempre aponta para a direção de **subida** mais íngreme. Portanto, para descer, nós simplesmente andamos na direção **oposta** a ele (`-∇f`).")

with st.expander("O que é uma Derivada Parcial? (Clique para expandir)"):
    st.markdown("""
    Não se assuste com o nome! Pense assim:
    - Você está na montanha, no ponto inicial (vermelho).
    - A **derivada parcial em relação a `x` (∂f/∂x)** mede a inclinação do terreno se você andar **apenas na direção Leste-Oeste** (eixo x).
    - A **derivada parcial em relação a `y` (∂f/∂y)** mede a inclinação se você andar **apenas na direção Norte-Sul** (eixo y).

    O **Vetor Gradiente `∇f`** é simplesmente a combinação dessas duas inclinações, nos dando a direção exata da subida mais íngreme.
    """)

st.markdown("As derivadas parciais da nossa função são:")
st.latex(r"\frac{\partial f}{\partial x} = y \cdot e^{-x^2 - y^2} \cdot (1 - 2x^2)")
st.latex(r"\frac{\partial f}{\partial y} = x \cdot e^{-x^2 - y^2} \cdot (1 - 2y^2)")
st.markdown("O Vetor Gradiente é:")
st.latex(r"\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)")

# --- Passo 3: A Caminhada (Iterações) ---
st.header(f"Passo 3: A Descida (Calculando {num_iteracoes} passo(s))")
st.markdown("Agora vamos simular a caminhada. A cada passo (iteração), nós seguimos a regra:")
st.latex(r"\text{ponto}_{\text{novo}} = \text{ponto}_{\text{antigo}} - \lambda \cdot \nabla f(\text{ponto}_{\text{antigo}})")

# Botão para iniciar o cálculo
if st.button("Iniciar a Descida!", type="primary"):

    # Realiza o algoritmo de descida do gradiente
    caminho = [ponto_inicial]
    ponto_atual = np.copy(ponto_inicial)

    for i in range(num_iteracoes):
        gradiente = grad_f(ponto_atual[0], ponto_atual[1])
        ponto_novo = ponto_atual - lambda_passo * gradiente
        caminho.append(ponto_novo)
        ponto_atual = ponto_novo

    caminho = np.array(caminho)

    # Exibe os cálculos passo a passo
    st.subheader("Detalhes dos Cálculos:")
    for i in range(num_iteracoes):
        with st.expander(f"Iteração {i+1} (Passo {i+1})"):
            p_antigo = caminho[i]
            p_novo = caminho[i+1]
            grad = grad_f(p_antigo[0], p_antigo[1])
            st.markdown(f"**Ponto de Partida (x_{i}):** `({p_antigo[0]:.4f}, {p_antigo[1]:.4f})`")
            st.markdown(f"**Calculando o Gradiente (∇f) neste ponto:** `({grad[0]:.4f}, {grad[1]:.4f})`")
            st.markdown(f"**Calculando o novo ponto (x_{i+1}):**")
            st.markdown(f"`({p_novo[0]:.4f}, {p_novo[1]:.4f}) = ({p_antigo[0]:.4f}, {p_antigo[1]:.4f}) - {lambda_passo} * ({grad[0]:.4f}, {grad[1]:.4f})`")


    # --- Passo 4: O Resultado Final ---
    st.header("Passo 4: Onde Chegamos?")
    ponto_final = caminho[-1]
    st.success(f"Após {num_iteracoes} passo(s), o ponto encontrado foi: **({ponto_final[0]:.4f}, {ponto_final[1]:.4f})**")
    st.info(f"O valor da função (altitude) neste ponto é: **{f(ponto_final[0], ponto_final[1]):.6f}**")


    # Plots Finais
    st.subheader("Visualizando o Caminho Percorrido")
    col1, col2 = st.columns(2)

    with col1:
        # Plot 3D com o caminho
        fig3d = matplotlib.pyplot.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, rcount=100, ccount=100)
        ax3d.plot(caminho[:, 0], caminho[:, 1], f(caminho[:, 0], caminho[:, 1]), 'r-o', markersize=4, label='Caminho da Descida')
        ax3d.scatter(ponto_final[0], ponto_final[1], f(ponto_final[0], ponto_final[1]), color='black', s=150, label='Ponto Final', zorder=10)
        ax3d.set_title("Caminho em 3D")
        ax3d.legend()
        st.pyplot(fig3d)

    with col2:
        # Plot 2D de contorno
        fig2d, ax2d = matplotlib.pyplot.subplots(figsize=(8, 6))
        cont = ax2d.contour(X, Y, Z, levels=20, cmap='viridis')
        ax2d.plot(caminho[:, 0], caminho[:, 1], 'r-o', markersize=4, label='Caminho da Descida')
        ax2d.scatter(ponto_inicial[0], ponto_inicial[1], c='blue', s=100, label='Ponto Inicial', ec='black')
        ax2d.scatter(ponto_final[0], ponto_final[1], c='black', s=100, label='Ponto Final')
        ax2d.set_title("Caminho em 2D (Mapa Topográfico)")
        ax2d.set_xlabel("Eixo X")
        ax2d.set_ylabel("Eixo Y")
        ax2d.legend()
        ax2d.grid(True)
        st.pyplot(fig2d)