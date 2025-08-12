### Entendendo o Conceito de Forma Fácil: A Caminhada na Montanha

Imagine que você está em uma montanha coberta por uma neblina densa e seu objetivo é chegar ao ponto mais baixo, o vale. Você não consegue ver o caminho todo, apenas o chão logo abaixo de seus pés. O que você faz?

1.  **Olhe ao seu redor:** Você sente a inclinação do terreno em todas as direções.
2.  **Escolha a direção:** Você identifica a direção que desce mais rápido, a mais íngreme.
3.  **Dê um passo:** Você dá um passo nessa direção.
4.  **Repita:** Agora, em sua nova posição, você repete o processo: olha ao redor, encontra a nova direção mais íngreme e dá mais um passo.

Você continua fazendo isso até chegar a um ponto onde o terreno é plano. Esse ponto é (provavelmente) o fundo de um vale, um **mínimo local**.

Isso é exatamente o que o **Algoritmo de Descida Mais Íngreme** (ou Gradiente Descendente) faz!

*   A **função `f(x,y)`** é a sua "montanha". O valor da função é a altitude.
*   O **ponto `(x, y)`** é a sua localização na montanha.
*   O **Gradiente (`∇f`)** é uma "bússola" que aponta para a direção em que a montanha sobe mais rápido (a direção mais íngreme para cima).
*   O **Gradiente Negativo (`-∇f`)** aponta para a direção oposta, ou seja, onde a montanha desce mais rápido. É essa a direção que seguimos!
*   O **Tamanho do Passo (`λ`)** é o quão grande é o passo que você dá a cada iteração.

O objetivo do exercício é usar essa "estratégia de caminhada" para encontrar o ponto mínimo da função `f(x,y) = xye^(-x² - y²)`.

---

### Resolvendo o Exercício Passo a Passo

Vamos agora aplicar esses conceitos ao seu problema.

**Dados do Problema:**
*   **Função:** `f(x,y) = xye^(-x²-y²)`
*   **Ponto Inicial:** `x₀ = (0.3, 1.2)`
*   **Tamanho do Passo:** `λ = 0.1`

#### 1. Cálculo do Gradiente (A nossa "Bússola")

Primeiro, precisamos encontrar o gradiente da função, `∇f(x,y)`. O gradiente é um vetor formado pelas derivadas parciais da função em relação a `x` e a `y`.

∇f(x,y) = ( ∂f/∂x , ∂f/∂y )

Para calcular as derivadas, usaremos a regra do produto `(uv)' = u'v + uv'`.

*   **Derivada em relação a x (`∂f/∂x`):**
    `∂f/∂x = [ (1) * y * e^(-x²-y²) ] + [ xy * e^(-x²-y²) * (-2x) ]`
    `∂f/∂x = y * e^(-x²-y²) * (1 - 2x²)`

*   **Derivada em relação a y (`∂f/∂y`):**
    `∂f/∂y = [ x * (1) * e^(-x²-y²) ] + [ xy * e^(-x²-y²) * (-2y) ]`
    `∂f/∂y = x * e^(-x²-y²) * (1 - 2y²)`

Portanto, nosso vetor gradiente é:
`∇f(x,y) = ( y * e^(-x²-y²) * (1 - 2x²) ,  x * e^(-x²-y²) * (1 - 2y²) )`

#### 2. Primeira Iteração (k=0)

Vamos dar o nosso primeiro passo a partir do ponto inicial `x₀ = (0.3, 1.2)`.

*   **Calcular o gradiente no ponto `x₀`:**
    Primeiro, vamos calcular o termo `e^(-x²-y²)`:
    `e^(-0.3² - 1.2²) = e^(-0.09 - 1.44) = e^(-1.53) ≈ 0.2165`

    Agora, as componentes do gradiente:
    `∂f/∂x = 1.2 * 0.2165 * (1 - 2 * 0.3²) = 0.2598 * (1 - 0.18) = 0.2598 * 0.82 ≈ 0.2130`
    `∂f/∂y = 0.3 * 0.2165 * (1 - 2 * 1.2²) = 0.0650 * (1 - 2.88) = 0.0650 * (-1.88) ≈ -0.1222`

    Então, `∇f(x₀) ≈ (0.2130, -0.1222)`.

*   **Atualizar o ponto (dar o passo):**
    A fórmula é: `x_novo = x_antigo - λ * ∇f(x_antigo)`
    `x₁ = x₀ - λ * ∇f(x₀)`
    `x₁ = (0.3, 1.2) - 0.1 * (0.2130, -0.1222)`
    `x₁ = (0.3, 1.2) - (0.0213, -0.0122)`
    `x₁ = (0.3 - 0.0213, 1.2 - (-0.0122))`
    `x₁ = (0.2787, 1.2122)`

**Ao final da primeira iteração, nosso novo ponto é `x₁ ≈ (0.2787, 1.2122)`.**

#### 3. Segunda Iteração (k=1)

Agora repetimos o processo, mas começando do nosso novo ponto `x₁`.

*   **Calcular o gradiente no ponto `x₁`:**
    Primeiro, o termo `e^(-x²-y²)` para `x₁`:
    `e^(-0.2787² - 1.2122²) = e^(-0.0777 - 1.4694) = e^(-1.5471) ≈ 0.2129`

    Agora, as componentes do gradiente:
    `∂f/∂x = 1.2122 * 0.2129 * (1 - 2 * 0.2787²) = 0.2581 * (1 - 0.1554) = 0.2581 * 0.8446 ≈ 0.2180`
    `∂f/∂y = 0.2787 * 0.2129 * (1 - 2 * 1.2122²) = 0.0593 * (1 - 2.9388) = 0.0593 * (-1.9388) ≈ -0.1149`

    Então, `∇f(x₁) ≈ (0.2180, -0.1149)`.

*   **Atualizar o ponto (dar o segundo passo):**
    `x₂ = x₁ - λ * ∇f(x₁)`
    `x₂ = (0.2787, 1.2122) - 0.1 * (0.2180, -0.1149)`
    `x₂ = (0.2787, 1.2122) - (0.0218, -0.0115)`
    `x₂ = (0.2787 - 0.0218, 1.2122 + 0.0115)`
    `x₂ = (0.2569, 1.2237)`

**Ao final da segunda iteração, nosso novo ponto é `x₂ ≈ (0.2569, 1.2237)`.**

---

### Guia para a Parte Computacional (Python)

Agora, vamos transformar essa lógica em um programa. Você pode usar o Google Colab, que é uma ferramenta online e gratuita que roda Python.

Aqui está um código em Python totalmente comentado para te guiar em cada um dos itens pedidos (a, b, c, d e o bônus).

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Definição das Funções ---
# Esta é a nossa "montanha"
def f(x, y):
    return x * y * np.exp(-x**2 - y**2)

# Esta é a nossa "bússola" que aponta para a subida mais íngreme
def grad_f(x, y):
    exp_term = np.exp(-x**2 - y**2)
    df_dx = y * exp_term * (1 - 2*x**2)
    df_dy = x * exp_term * (1 - 2*y**2)
    return np.array([df_dx, df_dy])

# --- Configurações Iniciais ---
ponto_inicial = np.array([0.3, 1.2])
lambda_passo = 0.1
erro_limite = 0.00001
eps = 1.0 # Erro inicial conforme solicitado

# --- Item (a): Plotar a função e o ponto inicial ---
print("--- Gerando Gráfico (a): Função e Ponto Inicial ---")
# Cria uma grade de pontos (x,y) para o plot
x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7) # A superfície da função

# Plota o ponto inicial
ax.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), color='red', s=100, label='Ponto Inicial (x₀)')

ax.set_title('Superfície da Função f(x,y) e Ponto Inicial')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.show()


# --- Item (b): Plotar o vetor gradiente na direção do mínimo ---
print("\n--- Gerando Gráfico (b): Vetor Gradiente ---")
# Calcula o gradiente no ponto inicial
gradiente_inicial = grad_f(ponto_inicial[0], ponto_inicial[1])
# A direção de descida é o gradiente negativo
direcao_descida = -gradiente_inicial

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]), color='red', s=100)

# Usa 'quiver' para desenhar a seta (vetor)
# O vetor começa em ponto_inicial e aponta na direção de descida
ax.quiver(ponto_inicial[0], ponto_inicial[1], f(ponto_inicial[0], ponto_inicial[1]),
          direcao_descida[0], direcao_descida[1], 0, # Z=0 pois é uma direção no plano xy
          length=0.5, normalize=True, color='black', label='Direção de Descida (-∇f)')

ax.set_title('Vetor Gradiente Apontando na Direção de Descida')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.show()

# --- Item (c): Rotina em Python para encontrar o mínimo ---
print(f"\n--- Executando Item (c): Rotina de Otimização ---")
ponto_atual = np.copy(ponto_inicial)
erro = eps
iteracao = 0
caminho = [ponto_atual] # Para o item bônus

print(f"Iniciando loop... Ponto inicial: {ponto_atual}, Erro inicial: {erro}")

while erro > erro_limite:
    f_atual = f(ponto_atual[0], ponto_atual[1])
    gradiente = grad_f(ponto_atual[0], ponto_atual[1])

    # A "mágica" do algoritmo: dar um passo na direção oposta ao gradiente
    ponto_novo = ponto_atual - lambda_passo * gradiente

    f_novo = f(ponto_novo[0], ponto_novo[1])
    erro = np.abs(f_novo - f_atual)

    # Atualiza para a próxima iteração
    ponto_atual = ponto_novo
    caminho.append(ponto_atual) # Guarda o ponto para vermos o caminho
    iteracao += 1

    if iteracao % 10 == 0: # Imprime o progresso a cada 10 passos
        print(f"Iteração {iteracao}: Ponto = {ponto_atual}, Erro = {erro}")

ponto_minimo = ponto_atual
print("\nLoop finalizado!")
print(f"Número de iterações: {iteracao}")
print(f"Ponto de mínimo encontrado: {ponto_minimo}")
print(f"Valor da função no mínimo: {f(ponto_minimo[0], ponto_minimo[1])}")


# --- Item (d): Plotar a função com o ponto de mínimo e o caminho ---
print("\n--- Gerando Gráfico (d): Resultado Final com Caminho ---")
caminho = np.array(caminho) # Converte a lista de pontos em um array numpy

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plota o caminho percorrido
ax.plot(caminho[:, 0], caminho[:, 1], f(caminho[:, 0], caminho[:, 1]), 'r-o', markersize=3, label='Caminho da Descida')
# Destaca o ponto final
ax.scatter(ponto_minimo[0], ponto_minimo[1], f(ponto_minimo[0], ponto_minimo[1]), color='black', s=150, label='Ponto Mínimo Encontrado', zorder=5)

ax.set_title('Função, Ponto Mínimo e Caminho Percorrido')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.show()

```
### Explicação Final do Código e dos Resultados

1.  **Gráfico (a):** Mostra a "montanha" (a função `f(x,y)` tem dois picos e dois vales) e onde você "começou a caminhada" (o ponto vermelho).
2.  **Gráfico (b):** Mostra a direção que o algoritmo escolheu para dar o primeiro passo. A seta preta aponta para a direção de descida mais íngreme a partir do ponto inicial.
3.  **Rotina (c):** O código executa exatamente os cálculos que fizemos manualmente, mas de forma repetida. Ele começa no ponto inicial, calcula o gradiente, dá um pequeno passo (`lambda = 0.1`) na direção oposta e repete. Ele para quando a "altitude" (o valor da função) quase não muda mais entre um passo e outro ( `erro < 0.00001` ).
4.  **Gráfico (d):** Este é o resultado final. Ele mostra o "caminho" (a linha vermelha) que o algoritmo percorreu, descendo a "encosta" da função, até chegar ao "vale" (o ponto preto), que é o mínimo local encontrado.

