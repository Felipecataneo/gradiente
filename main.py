import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os

# --- Configurações da Página ---
st.set_page_config(layout="wide", page_title="Descida do Gradiente - Análise Completa", page_icon="🏔️")

# --- Funções Matemáticas ---
def f(x, y):
    """Função objetivo: f(x,y) = xy * exp(-x² - y²)"""
    return x * y * np.exp(-x**2 - y**2)

def grad_f(x, y):
    """Gradiente da função f"""
    exp_term = np.exp(-x**2 - y**2)
    df_dx = y * exp_term * (1 - 2*x**2)
    df_dy = x * exp_term * (1 - 2*y**2)
    return np.array([df_dx, df_dy])

def algoritmo_gradiente_norma(ponto_inicial, lambda_passo, tolerancia=0.00001, max_iter=2000):
    """Executa o algoritmo de descida do gradiente usando a norma do gradiente como critério de parada."""
    ponto_atual = ponto_inicial.copy()
    caminho = [ponto_atual.copy()]
    historico_norma_grad = []
    historico_funcao = []
    
    for i in range(max_iter):
        gradiente = grad_f(ponto_atual[0], ponto_atual[1])
        norma_grad = np.linalg.norm(gradiente)
        
        historico_norma_grad.append(norma_grad)
        historico_funcao.append(f(ponto_atual[0], ponto_atual[1]))
        
        if norma_grad < tolerancia:
            break
            
        ponto_atual = ponto_atual - lambda_passo * gradiente
        caminho.append(ponto_atual.copy())
    
    return np.array(caminho), i, historico_norma_grad, historico_funcao

# <-- MUDANÇA: Ajustes no texto para remover subscritos
def create_pdf_report(params, resultados, paths_graficos):
    """Cria relatório em PDF com todos os resultados e gráficos."""
    
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "relatorio_gradiente.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=inch/2, bottomMargin=inch/2)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontSize=16, spaceAfter=20, alignment=1, textColor=colors.darkblue)
    story.append(Paragraph("Relatório - Algoritmo de Descida Mais Íngreme", title_style))
    
    story.append(Paragraph("Parâmetros da Simulação", styles['h2']))
    # Texto ajustado: "x0" e "y0" em vez de caracteres de subscrito
    params_text = f"""
    • Ponto inicial (x0, y0): ({params['x0']:.2f}, {params['y0']:.2f})<br/>
    • Tamanho do passo (λ): {params['lambda']}<br/>
    • Critério de parada: Norma do Gradiente &lt; {params['tolerancia']}
    """
    story.append(Paragraph(params_text, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Resolução Manual - Primeiras Iterações", styles['h2']))
    caminho_manual = resultados['caminho_manual']
    for i in range(min(2, len(caminho_manual) - 1)):
        p_atual, p_novo = caminho_manual[i], caminho_manual[i+1]
        grad = grad_f(p_atual[0], p_atual[1])
        story.append(Paragraph(f"<b>Iteração {i+1}</b>", styles['h3']))
        # Texto ajustado: removidas as tags <sub> para compatibilidade
        manual_text = f"""
        Ponto atual (x{i}, y{i}): ({p_atual[0]:.5f}, {p_atual[1]:.5f})<br/>
        Gradiente ∇f: [{grad[0]:.5f}, {grad[1]:.5f}]<br/>
        Novo ponto (x{i+1}, y{i+1}): ({p_novo[0]:.5f}, {p_novo[1]:.5f})
        """
        story.append(Paragraph(manual_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
    story.append(PageBreak())
    
    ponto_minimo = resultados['ponto_minimo']
    story.append(Paragraph("Resultado do Algoritmo Completo", styles['h2']))
    results_text = f"""
    • Convergência alcançada em: <b>{resultados['iteracoes']} iterações</b><br/>
    • Ponto de mínimo encontrado: ({ponto_minimo[0]:.6f}, {ponto_minimo[1]:.6f})<br/>
    • Valor mínimo da função: {f(ponto_minimo[0], ponto_minimo[1]):.8f}
    """
    story.append(Paragraph(results_text, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Gráficos da Execução", styles['h2']))
    story.append(Paragraph("Visão 3D e Curvas de Nível com o Caminho da Otimização:", styles['Normal']))
    story.append(Spacer(1, 12))
    
    img1 = Image(paths_graficos['caminho_3d'], width=3.5*inch, height=2.8*inch)
    img2 = Image(paths_graficos['caminho_contorno'], width=3.5*inch, height=2.8*inch)
    
    tabela_graficos1 = Table([[img1, img2]], colWidths=[3.6*inch, 3.6*inch])
    tabela_graficos1.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('PADDING', (0,0), (-1,-1), 0)]))
    story.append(tabela_graficos1)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Gráficos de Convergência:", styles['Normal']))
    story.append(Spacer(1, 12))
    img3 = Image(paths_graficos['convergencia_norma'], width=3.5*inch, height=2.8*inch)
    img4 = Image(paths_graficos['convergencia_funcao'], width=3.5*inch, height=2.8*inch)

    tabela_graficos2 = Table([[img3, img4]], colWidths=[3.6*inch, 3.6*inch])
    tabela_graficos2.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('PADDING', (0,0), (-1,-1), 0)]))
    story.append(tabela_graficos2)

    doc.build(story)
    return pdf_path

# --- Interface Streamlit (O restante do código permanece o mesmo) ---
st.title("🏔️ Algoritmo de Descida do Gradiente - Análise Completa")

st.sidebar.title("⚙️ Configurações")
x0_val = st.sidebar.slider("x₀ (Ponto Inicial)", -2.0, 2.0, 0.3, 0.1)
y0_val = st.sidebar.slider("y₀ (Ponto Inicial)", -2.0, 2.0, 1.2, 0.1)
lambda_passo = st.sidebar.slider("Tamanho do Passo (λ)", 0.01, 0.5, 0.1, 0.01)
tolerancia_val = st.sidebar.number_input("Tolerância (Norma do Gradiente)", 1e-6, 1e-2, 1e-5, format="%.6f")
num_iteracoes_manual = st.sidebar.slider("Iterações para Cálculo Manual", 1, 5, 2, 1)

current_params = {"x0": x0_val, "y0": y0_val, "lambda": lambda_passo, "tolerancia": tolerancia_val}
ponto_inicial = np.array([x0_val, y0_val])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizações Iniciais", "🔢 Cálculos Manuais", "🎯 Algoritmo Completo", "📄 Download PDF"])

x_grid, y_grid = np.linspace(-2.5, 2.5, 150), np.linspace(-2.5, 2.5, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

with tab1:
    st.header("Função, Ponto Inicial e Vetor Gradiente")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax1.scatter(ponto_inicial[0], ponto_inicial[1], f(*ponto_inicial), color='red', s=100, label='Ponto Inicial', zorder=10)
        ax1.set_title('Função Objetivo e Ponto Inicial')
        ax1.legend()
        st.pyplot(fig1)
    with col2:
        grad_inicial = grad_f(*ponto_inicial)
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.scatter(ponto_inicial[0], ponto_inicial[1], color='red', s=100, label='Ponto Inicial', zorder=5)
        ax2.quiver(ponto_inicial[0], ponto_inicial[1], -grad_inicial[0], -grad_inicial[1], 
                   color='black', scale=1, scale_units='xy', angles='xy', label='-∇f (Direção de Descida)')
        ax2.set_title('Curvas de Nível e Direção de Descida')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

with tab2:
    st.header(f"Cálculo das Primeiras {num_iteracoes_manual} Iterações")
    ponto_atual_manual, caminho_manual = ponto_inicial.copy(), [ponto_inicial.copy()]
    st.write(f"**Iteração 0:** Ponto = ({ponto_atual_manual[0]:.5f}, {ponto_atual_manual[1]:.5f}), f(x,y) = {f(*ponto_atual_manual):.6f}")
    for i in range(num_iteracoes_manual):
        grad = grad_f(*ponto_atual_manual)
        ponto_novo_manual = ponto_atual_manual - lambda_passo * grad
        caminho_manual.append(ponto_novo_manual)
        with st.expander(f"Detalhes da Iteração {i+1}"):
            st.latex(f"\\nabla f({ponto_atual_manual[0]:.4f}, {ponto_atual_manual[1]:.4f}) = \\begin{{bmatrix}} {grad[0]:.5f} \\\\ {grad[1]:.5f} \\end{{bmatrix}}")
            st.latex(f"x_{{{i+1}}} = {ponto_atual_manual[0]:.5f} - {lambda_passo} \\times ({grad[0]:.5f}) = {ponto_novo_manual[0]:.5f}")
            st.latex(f"y_{{{i+1}}} = {ponto_atual_manual[1]:.5f} - {lambda_passo} \\times ({grad[1]:.5f}) = {ponto_novo_manual[1]:.5f}")
        ponto_atual_manual = ponto_novo_manual
    st.session_state.caminho_manual_calculado = np.array(caminho_manual)

with tab3:
    st.header("Execução Completa do Algoritmo")
    if st.button("Executar Algoritmo até Convergência", type="primary"):
        with st.spinner("Calculando..."):
            caminho, iters, hist_norma, hist_func = algoritmo_gradiente_norma(ponto_inicial, lambda_passo, tolerancia_val)
            ponto_min = caminho[-1]
            st.session_state.resultados_execucao = {
                "caminho": caminho, "iteracoes": iters, "historico_norma": hist_norma,
                "historico_funcao": hist_func, "ponto_minimo": ponto_min,
                "caminho_manual": st.session_state.get('caminho_manual_calculado', [ponto_inicial])
            }
            st.session_state.parametros_execucao = current_params
            st.success(f"Convergência alcançada em {iters} iterações!")
            st.metric("Ponto de Mínimo (x, y)", f"({ponto_min[0]:.5f}, {ponto_min[1]:.5f})")
            st.metric("Valor Mínimo da Função", f"{f(*ponto_min):.7f}")

    if 'resultados_execucao' in st.session_state:
        res = st.session_state.resultados_execucao
        caminho, ponto_min, p_inicial_execucao = res['caminho'], res['ponto_minimo'], res['caminho'][0]

        st.subheader("Visualização do Caminho de Otimização")
        col1, col2 = st.columns(2)
        with col1:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, rcount=100, ccount=100)
            ax3.plot(caminho[:, 0], caminho[:, 1], f(caminho[:, 0], caminho[:, 1]), 'r-o', markersize=2)
            ax3.scatter(*p_inicial_execucao, f(*p_inicial_execucao), color='green', s=100, label='Início')
            ax3.scatter(*ponto_min, f(*ponto_min), color='red', s=150, label='Mínimo')
            ax3.set_title('Caminho de Otimização (3D)'); ax3.legend()
            st.pyplot(fig3)
        with col2:
            fig4, ax4 = plt.subplots()
            ax4.contour(X, Y, Z, levels=20, cmap='viridis')
            ax4.plot(caminho[:, 0], caminho[:, 1], 'r-o', markersize=2)
            ax4.scatter(*p_inicial_execucao, color='green', s=100, label='Início', zorder=5)
            ax4.scatter(*ponto_min, color='red', s=150, label='Mínimo', zorder=5)
            ax4.set_title('Caminho de Otimização (Curvas de Nível)'); ax4.legend()
            st.pyplot(fig4)
            
        st.subheader("Gráficos de Convergência")
        col3, col4 = st.columns(2)
        with col3:
            fig5, ax5 = plt.subplots()
            ax5.semilogy(res['historico_norma'])
            ax5.axhline(y=st.session_state.parametros_execucao['tolerancia'], color='r', linestyle='--')
            ax5.set_title('Convergência da Norma do Gradiente'); ax5.set_xlabel('Iteração'); ax5.set_ylabel('Norma (Log)')
            st.pyplot(fig5)
        with col4:
            fig6, ax6 = plt.subplots()
            ax6.plot(res['historico_funcao'])
            ax6.set_title('Evolução do Valor da Função'); ax6.set_xlabel('Iteração'); ax6.set_ylabel('f(x,y)')
            st.pyplot(fig6)
            
        temp_dir_imgs = tempfile.mkdtemp()
        paths = {k: os.path.join(temp_dir_imgs, f'{k}.png') for k in ['caminho_3d', 'caminho_contorno', 'convergencia_norma', 'convergencia_funcao']}
        fig3.savefig(paths['caminho_3d'], bbox_inches='tight'); fig4.savefig(paths['caminho_contorno'], bbox_inches='tight')
        fig5.savefig(paths['convergencia_norma'], bbox_inches='tight'); fig6.savefig(paths['convergencia_funcao'], bbox_inches='tight')
        st.session_state.paths_graficos = paths

with tab4:
    st.header("Download do Relatório em PDF")
    st.markdown("Gere um relatório completo em PDF com os parâmetros, cálculos e gráficos da **última simulação executada**.")

    if 'parametros_execucao' in st.session_state and st.session_state.parametros_execucao != current_params:
        st.warning("⚠️ Os parâmetros foram alterados. Execute o algoritmo novamente para gerar um relatório atualizado.")

    if st.button("Gerar Relatório PDF", type="primary", disabled=('resultados_execucao' not in st.session_state)):
        if 'resultados_execucao' not in st.session_state:
            st.error("Por favor, execute o algoritmo na aba 'Algoritmo Completo' primeiro.")
        else:
            with st.spinner("Gerando PDF..."):
                try:
                    pdf_path = create_pdf_report(
                        st.session_state.parametros_execucao,
                        st.session_state.resultados_execucao,
                        st.session_state.paths_graficos
                    )
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()

                    st.download_button(label="⬇️ Baixar Relatório", data=pdf_bytes, file_name="relatorio_descida_gradiente.pdf", mime="application/pdf")
                    os.unlink(pdf_path)
                    for path in st.session_state.paths_graficos.values(): os.unlink(path)
                except Exception as e:
                    st.error(f"Ocorreu um erro ao gerar o PDF: {e}")