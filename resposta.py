import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuração para evitar problemas com interface gráfica
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' se TkAgg não funcionar

# --- RESOLUÇÃO MANUAL DAS DUAS PRIMEIRAS ITERAÇÕES ---
def resolucao_manual():
    print("=" * 60)
    print("RESOLUÇÃO MANUAL - DUAS PRIMEIRAS ITERAÇÕES")
    print("=" * 60)
    
    # Função f(x,y) = xy * exp(-x² - y²)
    def f(x, y):
        return x * y * np.exp(-x**2 - y**2)
    
    # Gradiente: ∇f = [∂f/∂x, ∂f/∂y]
    # ∂f/∂x = y * exp(-x²-y²) * (1-2x²)
    # ∂f/∂y = x * exp(-x²-y²) * (1-2y²)
    def grad_f(x, y):
        exp_term = np.exp(-x**2 - y**2)
        df_dx = y * exp_term * (1 - 2*x**2)
        df_dy = x * exp_term * (1 - 2*y**2)
        return np.array([df_dx, df_dy])
    
    # Dados iniciais
    x0, y0 = 0.3, 1.2
    alpha = 0.1
    
    print(f"Ponto inicial: x₀ = {x0}, y₀ = {y0}")
    print(f"Tamanho do passo: α = {alpha}")
    print(f"f(x₀, y₀) = {f(x0, y0):.6f}")
    
    # ITERAÇÃO 1
    print("\n--- ITERAÇÃO 1 ---")
    grad_0 = grad_f(x0, y0)
    print(f"∇f(x₀, y₀) = [{grad_0[0]:.6f}, {grad_0[1]:.6f}]")
    
    x1 = x0 - alpha * grad_0[0]
    y1 = y0 - alpha * grad_0[1]
    print(f"x₁ = x₀ - α * ∂f/∂x = {x0} - {alpha} * {grad_0[0]:.6f} = {x1:.6f}")
    print(f"y₁ = y₀ - α * ∂f/∂y = {y0} - {alpha} * {grad_0[1]:.6f} = {y1:.6f}")
    print(f"f(x₁, y₁) = {f(x1, y1):.6f}")
    
    # ITERAÇÃO 2
    print("\n--- ITERAÇÃO 2 ---")
    grad_1 = grad_f(x1, y1)
    print(f"∇f(x₁, y₁) = [{grad_1[0]:.6f}, {grad_1[1]:.6f}]")
    
    x2 = x1 - alpha * grad_1[0]
    y2 = y1 - alpha * grad_1[1]
    print(f"x₂ = x₁ - α * ∂f/∂x = {x1:.6f} - {alpha} * {grad_1[0]:.6f} = {x2:.6f}")
    print(f"y₂ = y₁ - α * ∂f/∂y = {y1:.6f} - {alpha} * {grad_1[1]:.6f} = {y2:.6f}")
    print(f"f(x₂, y₂) = {f(x2, y2):.6f}")
    
    return (x0, y0), (x1, y1), (x2, y2)

# --- Definição das Funções ---
def f(x, y):
    """Função objetivo: f(x,y) = xy * exp(-x² - y²)"""
    return x * y * np.exp(-x**2 - y**2)

def grad_f(x, y):
    """Gradiente da função f"""
    exp_term = np.exp(-x**2 - y**2)
    df_dx = y * exp_term * (1 - 2*x**2)
    df_dy = x * exp_term * (1 - 2*y**2)
    return np.array([df_dx, df_dy])

# Executa a resolução manual
pontos_manuais = resolucao_manual()

# --- Configurações Iniciais ---
ponto_inicial = np.array([0.3, 1.2])
lambda_passo = 0.1
erro_limite = 0.00001
eps = 1.0

# --- Item (a): Plotar a função e o ponto inicial ---
print("\n" + "=" * 60)
print("ITEM (a): PLOTANDO FUNÇÃO E PONTO INICIAL")
print("=" * 60)

x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), 
          color='red', s=100, label='Ponto Inicial (x₀)')
ax.set_title('Superfície da Função f(x,y) = xy·exp(-x²-y²) e Ponto Inicial')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.savefig('grafico_a.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Item (b): Plotar o vetor gradiente na direção do mínimo ---
print("\nITEM (b): PLOTANDO VETOR GRADIENTE")
print("=" * 60)

gradiente_inicial = grad_f(ponto_inicial[0], ponto_inicial[1])
direcao_descida = -gradiente_inicial
print(f"Gradiente no ponto inicial: {gradiente_inicial}")
print(f"Direção de descida (-∇f): {direcao_descida}")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), 
          color='red', s=100, label='Ponto Inicial')

# Plota o vetor gradiente
ax.quiver(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]),
          direcao_descida[0], direcao_descida[1], 0,
          length=0.5, normalize=True, color='black', arrow_length_ratio=0.1,
          label='Direção de Descida (-∇f)', linewidth=3)

ax.set_title('Vetor Gradiente Apontando na Direção de Descida')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.savefig('grafico_b.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Item (c): Rotina em Python para encontrar o mínimo ---
print("\nITEM (c): EXECUTANDO ALGORITMO COMPLETO")
print("=" * 60)

ponto_atual = np.copy(ponto_inicial)
erro = eps
iteracao = 0
caminho = [ponto_atual.copy()]
historico_erro = []
historico_funcao = []

print(f"Ponto inicial: [{ponto_atual[0]:.3f}, {ponto_atual[1]:.3f}]")
print(f"Erro inicial: {erro}")
print(f"Critério de parada: erro < {erro_limite}")
print("\nIniciando otimização...")

while erro > erro_limite:
    f_atual = f(ponto_atual[0], ponto_atual[1])
    gradiente = grad_f(ponto_atual[0], ponto_atual[1])
    
    # Passo do gradiente descendente
    ponto_novo = ponto_atual - lambda_passo * gradiente
    f_novo = f(ponto_novo[0], ponto_novo[1])
    erro = np.abs(f_novo - f_atual)
    
    # Armazena histórico
    historico_erro.append(erro)
    historico_funcao.append(f_atual)
    
    # Atualiza
    ponto_atual = ponto_novo.copy()
    caminho.append(ponto_atual.copy())
    iteracao += 1
    
    if iteracao <= 10 or iteracao % 20 == 0:
        print(f"Iteração {iteracao:3d}: Ponto = [{ponto_atual[0]:8.5f}, {ponto_atual[1]:8.5f}], "
              f"f = {f_novo:10.6f}, Erro = {erro:.2e}")

ponto_minimo = ponto_atual
print(f"\nAlgoritmo convergiu após {iteracao} iterações!")
print(f"Ponto de mínimo: [{ponto_minimo[0]:.6f}, {ponto_minimo[1]:.6f}]")
print(f"Valor mínimo da função: {f(ponto_minimo[0], ponto_minimo[1]):.8f}")
print(f"Erro final: {erro:.2e}")

# --- Item (d): Plotar resultado final com caminho ---
print("\nITEM (d): PLOTANDO RESULTADO FINAL")
print("=" * 60)

caminho = np.array(caminho)

fig = plt.figure(figsize=(15, 10))

# Subplot 1: Gráfico 3D com caminho
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax1.plot(caminho[:, 0], caminho[:, 1], f(caminho[:, 0], caminho[:, 1]), 
         'r-o', markersize=2, linewidth=2, label='Caminho da Descida')
ax1.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), 
           color='green', s=100, label='Início')
ax1.scatter(ponto_minimo[0], ponto_minimo[1], f(ponto_minimo[0], ponto_minimo[1]), 
           color='red', s=150, label='Mínimo', zorder=5)
ax1.set_title('Função com Caminho de Otimização')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.legend()

# Subplot 2: Vista superior (contorno)
ax2 = fig.add_subplot(222)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(caminho[:, 0], caminho[:, 1], 'r-o', markersize=3, linewidth=2, 
         label='Caminho de Otimização')
ax2.scatter(ponto_inicial[0], ponto_inicial[1], color='green', s=100, label='Início', zorder=5)
ax2.scatter(ponto_minimo[0], ponto_minimo[1], color='red', s=100, label='Mínimo', zorder=5)
ax2.set_title('Vista Superior - Curvas de Nível')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Convergência do erro
ax3 = fig.add_subplot(223)
ax3.semilogy(range(1, len(historico_erro)+1), historico_erro, 'b-o', markersize=2)
ax3.set_title('Convergência do Erro')
ax3.set_xlabel('Iteração')
ax3.set_ylabel('Erro (log scale)')
ax3.grid(True, alpha=0.3)

# Subplot 4: Evolução do valor da função
ax4 = fig.add_subplot(224)
ax4.plot(range(len(historico_funcao)), historico_funcao, 'g-o', markersize=2)
ax4.axhline(y=f(ponto_minimo[0], ponto_minimo[1]), color='r', linestyle='--', 
           label=f'Valor mínimo = {f(ponto_minimo[0], ponto_minimo[1]):.6f}')
ax4.set_title('Evolução do Valor da Função')
ax4.set_xlabel('Iteração')
ax4.set_ylabel('f(x,y)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultado_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("RESUMO DOS RESULTADOS")
print("=" * 60)
print(f"✓ Função: f(x,y) = xy·exp(-x²-y²)")
print(f"✓ Ponto inicial: (0.3, 1.2)")
print(f"✓ Tamanho do passo: α = 0.1")
print(f"✓ Critério de parada: erro < {erro_limite}")
print(f"✓ Número de iterações: {iteracao}")
print(f"✓ Ponto de mínimo: ({ponto_minimo[0]:.6f}, {ponto_minimo[1]:.6f})")
print(f"✓ Valor mínimo: {f(ponto_minimo[0], ponto_minimo[1]):.8f}")
print(f"✓ Gráficos salvos como: grafico_a.png, grafico_b.png, resultado_final.png")
print("=" * 60)