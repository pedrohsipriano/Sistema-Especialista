# Sistema Especialista Híbrido — Inspeção de Peças por Braço Robótico

Trabalho da disciplina de **Inteligência Artificial e LFA** (UDF), Profa. Kadidja Valéria.

Sistema especialista que combina **aprendizado supervisionado** (árvore de decisão) com uma **base de regras SE/ENTÃO** para decidir o que um braço robótico deve fazer com cada peça numa linha de produção de automóveis.

> **Aviso:** todos os dados são **simulados** e foram criados apenas para demonstrar o método. O sistema não foi validado com peças ou especialistas reais.

---

## Sumário

1. [Problema](#problema)
2. [Como o sistema funciona](#como-o-sistema-funciona)
3. [Componentes do sistema especialista](#componentes-do-sistema-especialista)
4. [Dataset](#dataset)
5. [Modelo de aprendizado supervisionado](#modelo-de-aprendizado-supervisionado)
6. [Base de conhecimento (regras)](#base-de-conhecimento-regras)
7. [Origem do conhecimento](#origem-do-conhecimento)
8. [Como executar](#como-executar)
9. [Casos de teste](#casos-de-teste)
10. [Estudo inicial](#estudo-inicial)
11. [Limitações](#limitações)
12. [Melhorias futuras](#melhorias-futuras)
13. [Referências](#referências)

---

## Problema

No fim de uma linha de produção automotiva, cada peça (exemplo fictício: um motor de vidro elétrico) passa por uma inspeção. O sistema precisa:

1. **Classificar** a peça como `aprovada` ou `defeituosa` a partir de três medidas dos sensores;
2. **Decidir a ação** do braço robótico, levando em conta se os defeitos estão se repetindo na linha.

| Item | Descrição |
| --- | --- |
| **Domínio** | Robótica — controle de qualidade na indústria automotiva |
| **Público** | Operadores da linha de produção |
| **Entradas** | Peso (kg), comprimento (mm), ruído (dB) e se há defeito repetido |
| **Rótulo previsto** | `qualidade`: aprovada / defeituosa |
| **Saída** | Regra aplicada e ação do braço robótico |
| **Tipo de tarefa** | Interpretação (classificar a peça) + recomendação de ação |

---

## Como o sistema funciona

O sistema é **híbrido**: a previsão do modelo de Machine Learning vira um **fato** usado pelas regras do sistema especialista.

```
 Sensores                 FASE 1 (Machine Learning)            FASE 2 (Sistema Especialista)
┌──────────────┐        ┌───────────────────────────┐        ┌─────────────────────────────┐
│ peso_kg      │──────▶ │  Árvore de decisão         │──────▶ │  Motor de inferência         │──▶ Ação
│ comprimento  │        │  prevê: aprovada /         │ fato:  │  consulta a base de regras   │
│ ruido_db     │        │         defeituosa         │qualid. │  (R1 a R5)                   │
└──────────────┘        └───────────────────────────┘        └─────────────────────────────┘
                                                                     ▲
 Histórico da linha ── defeito_repetido (sim/não) ───────────────────┘
```

1. Os sensores medem peso, comprimento e ruído da peça.
2. **Fase 1:** a árvore de decisão prevê a `qualidade`.
3. O fato `defeito_repetido` informa se as peças anteriores também foram defeituosas.
4. **Fase 2:** o motor de inferência junta os dois fatos e escolhe a regra.
5. O sistema devolve a regra aplicada e a ação.

**Por que `defeito_repetido` não entra no modelo?** Porque ele não muda a qualidade *desta* peça — muda apenas *o que fazer* com ela. Compare R1 e R2: mesma previsão, ações diferentes. O modelo **classifica**; as regras **decidem**.

---

## Componentes do sistema especialista

| Componente | O que é | No código |
| --- | --- | --- |
| **Base de conhecimento** | Regras do domínio | Dicionário `regras` (R1–R4) + verificação da R5 |
| **Memória de trabalho** | Fatos do caso atual | Medidas da peça, `qualidade` prevista e `defeito_repetido` |
| **Motor de inferência** | Aplica as regras aos fatos | Função `inspecionar()` |
| **Modelo supervisionado** | Gera o fato `qualidade` | `DecisionTreeClassifier` |

**Estratégia de inferência:** encadeamento **para frente** — parte dos fatos (medidas) e chega à conclusão (ação).

---

## Dataset

30 peças **simuladas**, definidas diretamente no código.

| Grupo | Quantidade | Característica | Rótulo |
| --- | --- | --- | --- |
| Peças boas | 15 | ~1,20 kg, 148–151 mm, 36–44 dB | `aprovada` |
| Peças leves | 5 | peso entre 1,06 e 1,13 kg (falta material) | `defeituosa` |
| Peças compridas | 5 | comprimento entre 153 e 156 mm (rebarba) | `defeituosa` |
| Peças barulhentas | 5 | ruído entre 47 e 55 dB (engrenagem/rolamento ruim) | `defeituosa` |

**Critério de rotulagem** (representa a decisão de um inspetor humano):

- `defeituosa` se peso < 1,15 kg **ou** comprimento > 152 mm **ou** ruído > 45 dB;
- `aprovada` caso contrário.

Esses limites servem apenas para **rotular** os dados simulados. O modelo **não** recebe esses números — ele os aprende a partir dos exemplos.

| Atributo | Tipo | Unidade |
| --- | --- | --- |
| `peso_kg` | número decimal | quilogramas |
| `comprimento_mm` | número | milímetros |
| `ruido_db` | número | decibéis |
| `qualidade` (rótulo) | categoria | aprovada / defeituosa |

---

## Modelo de aprendizado supervisionado

- **Algoritmo:** `DecisionTreeClassifier` (scikit-learn), `random_state=42`.
- **Divisão:** `train_test_split` com `test_size=0.25`, `stratify=y` e `random_state=42` → **22 peças de treino** e **8 de teste** (4 aprovadas e 4 defeituosas).
  - `stratify=y` mantém a proporção das classes no treino e no teste;
  - `random_state=42` deixa a divisão reproduzível.

### Resultado

```
Acurácia: 1.0
[[4 0]
 [0 4]]
```

| | Previsto: aprovada | Previsto: defeituosa |
| --- | --- | --- |
| **Real: aprovada** | 4 | 0 |
| **Real: defeituosa** | 0 | 4 |

A diagonal são os acertos. O erro mais grave seria o canto inferior esquerdo (peça defeituosa aprovada), que iria para o carro.

### Condições aprendidas pela árvore

Obtidas com `export_text(modelo, feature_names=list(X.columns))`:

```
|--- ruido_db <= 45.00
|   |--- peso_kg <= 1.15
|   |   |--- class: defeituosa
|   |--- peso_kg >  1.15
|   |   |--- comprimento_mm <= 152.50
|   |   |   |--- class: aprovada
|   |   |--- comprimento_mm >  152.50
|   |   |   |--- class: defeituosa
|--- ruido_db >  45.00
|   |--- class: defeituosa
```

A árvore descobriu sozinha limites muito próximos aos do critério de rotulagem. **Essas condições vêm dos dados; as regras R1–R5 foram escritas a partir do conhecimento do domínio.**

> **Atenção:** 100% de acurácia em 8 casos simulados **não** demonstra eficácia real. O teste é pequeno e as classes foram construídas para serem fáceis de separar.

---

## Base de conhecimento (regras)

| Regra | SE | ENTÃO |
| --- | --- | --- |
| **R5** (verificada primeiro) | alguma medida está ausente | chamar o operador para inspeção manual |
| **R1** | `qualidade = defeituosa` E `defeito_repetido = não` | descartar a peça |
| **R2** | `qualidade = defeituosa` E `defeito_repetido = sim` | parar a linha e chamar o operador |
| **R3** | `qualidade = aprovada` E `defeito_repetido = sim` | montar e avisar o operador para revisar a máquina |
| **R4** | `qualidade = aprovada` E `defeito_repetido = não` | enviar para a montagem |

- R1–R4 cobrem todas as combinações de `qualidade` × `defeito_repetido`, então nunca há conflito nem falta de regra.
- A R5 vem antes do modelo: **ausência de evidência não é negação**. Não saber o ruído é diferente de saber que ele está ok, então o sistema não decide sozinho e encaminha para um humano.

---

## Origem do conhecimento

Sem um inspetor de qualidade disponível, o conhecimento foi adquirido por **análise de documentos e protocolos**.

| Fonte | O que sustenta |
| --- | --- |
| **IATF 16949** — norma de qualidade automotiva | R1 (produto não conforme não segue) e R5 (medição precisa ser confiável) |
| **Jidoka** (Sistema Toyota de Produção) — detectar, parar, corrigir, investigar | R2 (parar a linha e chamar o operador) e R5 (chamar ajuda humana) |
| **Regras de Western Electric** (controle estatístico de processo) | `defeito_repetido`, R2 e R3: padrões repetidos indicam processo fora de controle |
| **Testes de fim de linha** (Siemens, HBK) | Atributo `ruido_db`, decisão aprovado/reprovado e uso de dados para refinar limites |

O "3 peças defeituosas seguidas" para `defeito_repetido` é uma simplificação de projeto inspirada nas regras de Western Electric. Numa versão real, um inspetor de qualidade validaria as regras.

---

## Como executar

### Requisitos

- Python 3.9+
- Bibliotecas: `pandas`, `scikit-learn`

### Passo a passo

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Instalar as dependências
pip install pandas scikit-learn

# 3. Executar (as aspas são necessárias porque o nome tem espaços)
python "A1 - Sistema Especialista.py"
```

> No VS Code, se aparecer `ModuleNotFoundError: No module named 'sklearn'`, selecione o interpretador do ambiente virtual: **Ctrl + Shift + P → Python: Select Interpreter → `.venv`**.

---

## Casos de teste

Seguindo a Aula 12, o sistema é testado com um caso **positivo**, um **negativo** e um de **informação ausente**, além de casos de borda: medidas fora da faixa vista no treino e ruído logo acima do limite.

```python
print(inspecionar(1.20, 150, 40, True))    # caso 1
print(inspecionar(1.10, 150, 40, True))    # caso 2
print(inspecionar(1.00, 151, 20, True))    # caso 3
print(inspecionar(1.19, 151, 35, True))    # caso 4
print(inspecionar(1.19, 151, 40, True))    # caso 5
print(inspecionar(1.19, 170, None, True))  # caso 6
print(inspecionar(1.19, 151, 46, True))    # caso 7
```

| Caso | Entradas | Previsão (ML) | Regra | Ação | Por quê |
| --- | --- | --- | --- | --- | --- |
| 1. Positivo | 1,20 kg, 150 mm, 40 dB, com repetição | aprovada | R3 | Montar e avisar o operador | Medidas normais, mas a linha mostra defeitos repetidos |
| 2. Negativo | 1,10 kg, 150 mm, 40 dB, com repetição | defeituosa | R2 | Parar a linha e chamar o operador | Peça leve demais (peso ≤ 1,15 kg) e defeitos seguidos indicam máquina desregulada |
| 3. Fora do treino | 1,00 kg, 151 mm, 20 dB, com repetição | defeituosa | R2 | Parar a linha e chamar o operador | Peso ≤ 1,15 kg; peso e ruído estão fora da faixa vista no treino (ver Limitações) |
| 4. Ruído baixo | 1,19 kg, 151 mm, 35 dB, com repetição | aprovada | R3 | Montar e avisar o operador | Ruído abaixo de 45 dB (embora menor que o mínimo do treino, 36 dB) e demais medidas normais |
| 5. Aprovada | 1,19 kg, 151 mm, 40 dB, com repetição | aprovada | R3 | Montar e avisar o operador | A peça está boa, mas a linha mostra sinais de problema |
| 6. Informação ausente | 1,19 kg, 170 mm, ruído não medido, com repetição | — | R5 | Chamar o operador para inspeção manual | Sem o ruído o sistema não conclui nada; a R5 vem antes do modelo, mesmo com o comprimento de 170 mm fora do normal |
| 7. Ruído no limite | 1,19 kg, 151 mm, 46 dB, com repetição | defeituosa | R2 | Parar a linha e chamar o operador | Ruído acima de 45 dB, o limite que a árvore aprendeu |

> Todos os casos usam `defeito_repetido = True`, por isso as regras R1 e R4 não aparecem nesta execução.

### Saída esperada

```
Acurácia: 1.0
[[4 0]
 [0 4]]
('R3', 'Montar e avisar o operador para revisar a máquina.')
('R2', 'Parar a linha e chamar o operador.')
('R2', 'Parar a linha e chamar o operador.')
('R3', 'Montar e avisar o operador para revisar a máquina.')
('R3', 'Montar e avisar o operador para revisar a máquina.')
('R5', 'Chamar o operador para inspeção manual.')
('R2', 'Parar a linha e chamar o operador.')
```

---

## Estudo inicial

**1. O que é aprendizado de máquina?**
É quando um algoritmo ajusta um modelo a partir de dados, em vez de uma pessoa escrever as regras à mão.

**2. O que caracteriza o aprendizado supervisionado?**
O modelo aprende com exemplos que já vêm com a resposta certa (o rótulo), como um gabarito, e usa os padrões aprendidos para prever casos novos. Neste sistema, cada peça de treino já vem marcada como aprovada ou defeituosa.

**3. O que são atributos e rótulos?**
Atributos são as entradas que descrevem o caso: `peso_kg`, `comprimento_mm` e `ruido_db`. Rótulo é a resposta conhecida: `qualidade` (aprovada/defeituosa).

**4. Como classificação difere de regressão?**
Classificação prevê uma categoria, como aprovada ou defeituosa. Regressão prevê um número, como quantos dB a peça vai fazer. Este sistema faz classificação.

**5. Por que separar treino e teste?**
Para medir o modelo em casos que ele nunca viu. Avaliando com os mesmos dados do treino, ele pode só ter "decorado" e parecer melhor do que é. Aqui são 22 peças para treino e 8 para teste.

---

## Limitações

- **Dados simulados e poucos:** 30 peças criadas para separar facilmente as classes; o resultado não representa uma fábrica real.
- **Teste pequeno:** 100% de acurácia em 8 casos exige cautela.
- **Só três atributos:** o sistema não detecta trincas, cor errada ou peça trocada.
- **Valores fora do treino:** uma peça com medidas muito diferentes das vistas no treino (ex.: 1,00 kg ou 20 dB) ainda é classificada pelo modelo, que pode errar fora do domínio de treinamento.
- **`defeito_repetido` informado manualmente:** o histórico da linha não é calculado pelo sistema.
- **Sem especialista real:** as regras vêm de documentos e não foram validadas por um inspetor de qualidade.
- **Supervisão humana:** a recomendação final deve sempre permitir explicação e revisão humana.

---

## Melhorias futuras

- [ ] Estender a R5 para medidas **fora do intervalo** visto no treino.
- [ ] Mostrar as condições aprendidas pela árvore (`export_text` / `plot_tree`).
- [ ] Retornar uma **explicação** junto da regra (fatos usados e motivo).
- [ ] Calcular `defeito_repetido` automaticamente a partir do histórico das últimas peças.
- [ ] Salvar o dataset em CSV (`dados.to_csv("pecas.csv", index=False)`).
- [ ] Usar mais dados e validação separada para avaliar melhor o modelo.

---

## Referências

- Profa. Kadidja Valéria. *Sistemas Especialistas: conhecimento, regras e inferência* (Aula 12). UDF.
- Profa. Kadidja Valéria. *Aprendizado de Máquina — Introdução* (Aula 13). UDF.
- Profa. Kadidja Valéria. *Aprendizado Supervisionado em Sistemas Especialistas* (apresentação). UDF.
- Lenovo. [Expert Systems: A Comprehensive Guide](https://www.lenovo.com/pt/pt/knowledgebase/expert-systems-a-comprehensive-guide/).
- IBM. Guia de Aprendizado de Máquina.
- DQS. [IATF 16949 explicada](https://www.dqsglobal.com/pt/explore/centro-de-conhecimento-dqs/iatf-16949-explicada).
- Kaizen Institute. [Jidoka: a base da automação com um toque humano](https://kaizen.com/pt/insights-pt/jidoka-automacao-toque-humano/).
- [Western Electric rules](https://en.wikipedia.org/wiki/Western_Electric_rules).
- Siemens. [End of Line Quality Testing — Simcenter Anovis](https://community.sw.siemens.com/articles/en_US/Knowledge/End-of-Line-Quality-Testing-Simcenter-Anovis).
- HBK. [Testes de produção de componentes automotivos](https://hbkworld.com/pt/solutions/applications/asset-process-monitoring/production-testing/automotive-components).
- scikit-learn. [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) e [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).