# app.py
"""
App Streamlit: Visualizador de árboles generadores en K(m,n)

Características:
- Cálculo de τ(K_{m,n}) = m^{n-1} n^{m-1}
- Ejemplos (hasta 10 árboles generadores, o todos si el número es pequeño)
- Animación Árbol → Sucesión (Hartsfield–Werth)
- Animación Sucesión → Árbol (proceso inverso)
- Panel teórico y parámetros (semilla aleatoria, velocidad, etc.)

Para ejecutar:
    streamlit run app.py
"""

from __future__ import annotations
import time
from typing import List, Tuple

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from logic import (
    build_complete_bipartite,
    build_tree_graph,
    count_spanning_trees,
    random_spanning_tree_bipartite,
    tree_to_sequence_steps,
    sequence_to_tree_steps,
    Label,
    Edge,
)


# -----------------------------
# Auxiliar de dibujo
# -----------------------------

def draw_bipartite_tree(
    G: nx.Graph,
    M: List[Label],
    N: List[Label],
    highlight_node: Label | None = None,
):
    """
    Dibuja el árbol bipartito con M a la izquierda y N a la derecha.
    Si highlight_node se pasa, se resalta ese nodo.
    """
    pos = {}
    for i, a in enumerate(M):
        pos[a] = (0, i)
    for j, b in enumerate(N):
        pos[b] = (1, j)

    fig, ax = plt.subplots()
    node_colors = []
    for n in G.nodes():
        if n in M:
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightblue")

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        ax=ax,
    )

    if highlight_node is not None and highlight_node in G.nodes():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[highlight_node],
            node_color="yellow",
            node_size=700,
            edgecolors="black",
            ax=ax,
        )

    ax.set_axis_off()
    fig.tight_layout()
    return fig


# -----------------------------
# Configuración Streamlit
# -----------------------------

st.set_page_config(
    page_title="K(m,n): Árboles Generadores",
    layout="wide",
)

st.title("🌳 Árboles generadores en el grafo bipartito completo K(m,n)")

st.markdown(
    """
Esta aplicación está inspirada en el artículo  
**"Spanning Trees of the Complete Bipartite Graph" de Hartsfield & Werth**.

Aquí puedes:

1. Explorar cuántos árboles generadores tiene \(K_{m,n}\).
2. Ver ejemplos de árboles generadores.
3. Ver una **animación de cómo se construye la sucesión** a partir de un árbol (Árbol → Sucesión).
4. Ver la **animación inversa**: cómo se reconstruye el árbol a partir de la sucesión (Sucesión → Árbol).
5. Revisar un pequeño resumen teórico del resultado.
"""
)

# Sidebar: parámetros globales
st.sidebar.header("Parámetros de K(m,n)")

m = st.sidebar.slider("m (vértices en M)", min_value=2, max_value=6, value=3)
n = st.sidebar.slider("n (vértices en N)", min_value=2, max_value=6, value=3)

# Semilla para reproducibilidad
seed = st.sidebar.number_input(
    "Semilla aleatoria (0 = sin semilla fija)",
    min_value=0,
    value=0,
    step=1,
)
if seed != 0:
    import random
    random.seed(seed)

M, N = build_complete_bipartite(m, n)

st.sidebar.markdown(
    f"**Conjuntos:**  \n"
    f"M = {{ {', '.join(M)} }}  \n"
    f"N = {{ {', '.join(N)} }}"
)

# Número total de árboles
total_trees = count_spanning_trees(m, n)

st.subheader("Número total de árboles generadores de K(m,n)")
st.latex(r"\tau(K_{m,n}) = m^{n-1} \cdot n^{m-1}")
st.markdown(
    f"Para m = **{m}**, n = **{n}**:  \n"
    f"\\( \\tau(K_{{{m},{n}}}) = {m}^{{{n-1}}} \\cdot {n}^{{{m-1}}} = {total_trees} \\)"
)

st.info(
    "El número de árboles crece muy rápido. Para tamaños pequeños podemos intentar "
    "mostrar casi todos; para tamaños grandes, solo unos pocos ejemplos aleatorios."
)

# Velocidad de animación
speed = st.sidebar.slider(
    "Velocidad de animación (segundos por paso)",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1,
)

# Tabs principales
tab_intro, tab_examples, tab_anim_forward, tab_anim_inverse, tab_theory = st.tabs(
    [
        "Introducción",
        "Ejemplos de árboles",
        "Animación Árbol → Sucesión",
        "Animación Sucesión → Árbol",
        "Teoría",
    ]
)

# -----------------------------
# TAB 1: Introducción
# -----------------------------

with tab_intro:
    st.markdown("## 🧩 Introducción")

    st.markdown(
        """
El grafo **bipartito completo** \(K_{m,n}\) tiene:

- Un conjunto de vértices \\(M = \\{a_1, \\dots, a_m\\}\\)
- Un conjunto de vértices \\(N = \\{b_1, \\dots, b_n\\}\\)
- Todas las aristas posibles entre M y N, pero **ninguna** dentro de M o dentro de N.

Un **árbol generador** es un subgrafo:
- conexo
- sin ciclos
- que contiene **todos los vértices** del grafo original.

El resultado clásico dice que:

\\[
\\tau(K_{m,n}) = m^{n-1}\\, n^{m-1}
\\]

En esta app puedes **ver** ese resultado en acción y entender cómo aparece
la correspondencia entre árboles y sucesiones, igual que en el artículo de Hartsfield & Werth.
"""
    )

# -----------------------------
# TAB 2: Ejemplos de árboles
# -----------------------------

with tab_examples:
    st.markdown("## 🌲 Ejemplos de árboles generadores")

    # límite razonable para intentar cubrir "todos"
    max_show_all = 20
    num_samples = min(10, total_trees)

    if total_trees <= max_show_all:
        st.markdown(
            f"Como \\(\\tau(K_{{{m},{n}}}) = {total_trees} \\le {max_show_all}\\), "
            "intentaremos mostrar **todos** los árboles (si los podemos generar)."
        )
        target = total_trees
    else:
        st.markdown(
            f"El número total de árboles es **{total_trees}**, muy grande para verlos todos.  \n"
            f"Mostramos **{num_samples}** ejemplos aleatorios."
        )
        target = num_samples

    sampled_edge_sets = set()
    sampled_trees: List[List[Edge]] = []

    max_tries = 1000 * target
    tries = 0

    # Intentamos recolectar hasta 'target' árboles diferentes vía random
    while len(sampled_trees) < target and tries < max_tries:
        tries += 1
        edges = random_spanning_tree_bipartite(M, N)
        normalized = tuple(sorted(tuple(sorted(e)) for e in edges))
        if normalized not in sampled_edge_sets:
            sampled_edge_sets.add(normalized)
            sampled_trees.append(edges)

    if total_trees <= max_show_all and len(sampled_trees) < total_trees:
        st.warning(
            "Se intentó generar todos los árboles, pero es posible que no se hayan "
            "encontrado absolutamente todos (generación aleatoria)."
        )

    cols = st.columns(2)

    for idx, edges in enumerate(sampled_trees, start=1):
        G_tree = build_tree_graph(M, N, edges)
        fig = draw_bipartite_tree(G_tree, M, N)
        col = cols[(idx - 1) % 2]
        with col:
            st.markdown(f"**Árbol {idx}**")
            st.pyplot(fig)


# -----------------------------
# TAB 3: Animación Árbol → Sucesión
# -----------------------------

with tab_anim_forward:
    st.markdown("## 🔁 Animación: Árbol → Sucesión")

    st.markdown(
        """
Este modo muestra cómo, a partir de un árbol generador, se va construyendo
la sucesión eliminando hojas y anotando al vecino, como en el artículo de Hartsfield & Werth.
"""
    )

    if st.button("🎬 Generar árbol y ver animación Árbol → Sucesión"):
        # Generamos un árbol aleatorio y obtenemos sus pasos
        sim_edges = random_spanning_tree_bipartite(M, N)
        steps, seq = tree_to_sequence_steps(M, N, sim_edges)

        # Guardamos info para usar luego en la animación inversa
        st.session_state.last_seq = seq
        st.session_state.last_M = M
        st.session_state.last_N = N
        st.session_state.last_edges = sim_edges

        placeholder_plot = st.empty()
        placeholder_text = st.empty()
        progress_bar = st.progress(0.0)

        removed_so_far = set()

        total_steps = len(steps)

        for i, step in enumerate(steps, start=1):
            removed_so_far.add(step["removed"])
            G_work = build_tree_graph(M, N, sim_edges)
            for r in removed_so_far:
                if r in G_work.nodes:
                    G_work.remove_node(r)

            fig = draw_bipartite_tree(G_work, M, N, highlight_node=step["neighbor"])
            placeholder_plot.pyplot(fig)

            placeholder_text.markdown(
                f"**Paso {i}/{total_steps}**  \n"
                f"- Nodo eliminado: `{step['removed']}` (en {step['chosen_side']})  \n"
                f"- Vecino anotado en la sucesión: `{step['neighbor']}`  \n"
                f"- Sucesión parcial: `{step['sequence']}`"
            )

            progress_bar.progress(i / total_steps)
            time.sleep(speed)

        st.success(f"✅ Sucesión completa generada: `{seq}`")


# -----------------------------
# TAB 4: Animación Sucesión → Árbol
# -----------------------------

with tab_anim_inverse:
    st.markdown("## 🔁 Animación: Sucesión → Árbol")

    st.markdown(
        """
Aquí se ve el proceso inverso: dada una sucesión (como la generada en la pestaña anterior),
se reconstruye paso a paso el árbol que le corresponde.
"""
    )

    col_input, col_btn = st.columns([2, 1])

    # Permitir usar la última sucesión o escribir una manual
    use_last = False
    with col_input:
        manual_seq_str = st.text_input(
            "Sucesión (opcional, separada por comas, por ejemplo: a1,a2,b1,a2)",
            value="",
            help="Si la dejas vacía, se usará la última sucesión generada en la pestaña anterior.",
        )

    with col_btn:
        if st.button("🎬 Ver animación Sucesión → Árbol"):
            # Decidir qué sucesión usar
            if manual_seq_str.strip():
                seq_labels = [s.strip() for s in manual_seq_str.split(",") if s.strip()]
                seq = seq_labels
                M_for_seq = M
                N_for_seq = N
            else:
                if "last_seq" not in st.session_state:
                    st.warning(
                        "No hay sucesión previa. Escribe una sucesión manual arriba "
                        "o primero genera una en 'Árbol → Sucesión'."
                    )
                    seq = None
                else:
                    seq = st.session_state.last_seq
                    M_for_seq = st.session_state.last_M
                    N_for_seq = st.session_state.last_N

            if seq is not None:
                try:
                    steps_inv, full_edges_inv = sequence_to_tree_steps(M_for_seq, N_for_seq, seq)
                except Exception as e:
                    st.error(f"Ocurrió un error reconstruyendo el árbol: {e}")
                else:
                    placeholder_plot2 = st.empty()
                    placeholder_text2 = st.empty()
                    progress_bar2 = st.progress(0.0)

                    total_steps_inv = len(steps_inv)

                    for i, step in enumerate(steps_inv, start=1):
                        edges_so_far = step["edges_so_far"]
                        new_edge = step["edge"]

                        G_partial = build_tree_graph(M_for_seq, N_for_seq, edges_so_far)
                        # resaltar el vértice recién agregado (segundo del par)
                        highlight = new_edge[1]

                        fig2 = draw_bipartite_tree(
                            G_partial, M_for_seq, N_for_seq, highlight_node=highlight
                        )
                        placeholder_plot2.pyplot(fig2)

                        placeholder_text2.markdown(
                            f"**Paso {i}/{total_steps_inv}**  \n"
                            f"- Arista agregada: `{new_edge}`  \n"
                            f"- Número de aristas actuales: {len(edges_so_far)}"
                        )

                        progress_bar2.progress(i / total_steps_inv)
                        time.sleep(speed)

                    st.success("✅ Reconstrucción completa del árbol a partir de la sucesión.")


# -----------------------------
# TAB 5: Teoría
# -----------------------------

with tab_theory:
    st.markdown("## 📚 Resumen teórico")

    st.markdown(
        r"""
**Teorema (Hartsfield–Werth, caso bipartito):**  
El número de árboles generadores de \(K_{m,n}\) es

\[
\tau(K_{m,n}) = m^{n-1}\, n^{m-1}.
\]

La demostración se basa en construir una biyección entre:

1. Los árboles generadores de \(K_{m,n}\), y  
2. Ciertas sucesiones (códigos) de longitud \(m + n - 2\) formadas por vértices de \(M \cup N\).

---

### Idea de Árbol → Sucesión

1. Partimos de un árbol generador \(T\).
2. Mientras el árbol tenga más de 2 vértices:
   - Buscamos una **hoja** en el lado \(N\) (si existe) con menor subíndice.
   - Si no hay hojas en \(N\), buscamos una hoja en \(M\).
   - Anotamos en la sucesión el **vecino** de esa hoja.
   - Eliminamos la hoja del árbol.
3. Al final obtenemos una sucesión de longitud \(m + n - 2\).

Este procedimiento es inyectivo (no colapsa dos árboles en la misma sucesión).

---

### Idea de Sucesión → Árbol

El proceso inverso toma una sucesión y:

1. Reconstruye las aristas, eligiendo en cada paso el vértice de la otra partición
   que **no vuelve a aparecer** en la sucesión.
2. Al final se conecta el último vértice restante de \(M\) con el último de \(N\).

Ese proceso es la inversa de Árbol → Sucesión, así que se tiene una biyección.

---

### ¿Por qué \(m^{n-1} n^{m-1}\)?

En el artículo original, se muestra que el número de sucesiones válidas es:

\[
m^{n-1} \, n^{m-1},
\]

y como hay una correspondencia 1–1 entre árboles y sucesiones,
ese es también el número de árboles generadores de \(K_{m,n}\).

Esta app está pensada para que **veas esa biyección en acción** con ejemplos
y animaciones 😄.
"""
    )
