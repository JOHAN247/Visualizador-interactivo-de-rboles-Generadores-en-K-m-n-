# app.py
"""
App Streamlit: Visualizador de árboles generadores en K(m,n)

Mejoras (según retroalimentación):
1) Árbol → Sucesión: grafo pequeño y fijo a la derecha para apreciar la animación.
2) Sucesión → Árbol: instrucciones claras sobre sucesiones válidas (longitud, tipo, conteos),
   explicación de validez en tiempo real y pasos detallados.
3) Ejemplos: cada árbol muestra un icono ❓ con su sucesión asociada.
4) Sucesión → Árbol: ahora hay un selector de sucesiones válidas (tipo "semilla"),
   al elegir una se explica por qué es válida y luego se puede iniciar la animación.

Para ejecutar:
    streamlit run app.py
"""

from __future__ import annotations
import time
import re
import random
from typing import List, Tuple, Dict, Any

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
# Utilidades UI
# -----------------------------

def parse_sequence(text: str) -> List[str]:
    """Parsea sucesión aceptando comas o espacios: 'a1,a2 b1' -> ['a1','a2','b1']"""
    text = text.strip().replace("[", "").replace("]", "")
    if not text:
        return []
    parts = re.split(r"[,\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def validate_sequence(seq: List[Label], M: List[Label], N: List[Label]) -> Tuple[bool, Dict[str, Any]]:
    """
    Valida sucesión para el modelo de la app:
    - Longitud exacta: m+n-2
    - Símbolos permitidos: M ∪ N
    - Conteos exactos: (n-1) símbolos en M y (m-1) símbolos en N
    """
    m = len(M)
    n = len(N)
    expected_len = m + n - 2

    allowed = set(M) | set(N)
    bad = [x for x in seq if x not in allowed]

    count_M = sum(1 for x in seq if x in M)
    count_N = sum(1 for x in seq if x in N)

    reasons = {
        "expected_len": expected_len,
        "actual_len": len(seq),
        "allowed_ok": len(bad) == 0,
        "bad_symbols": bad,
        "expected_count_M": n - 1,
        "expected_count_N": m - 1,
        "count_M": count_M,
        "count_N": count_N,
    }

    ok = True
    if len(seq) != expected_len:
        ok = False
    if bad:
        ok = False
    if count_M != (n - 1) or count_N != (m - 1):
        ok = False

    return ok, reasons


def generate_valid_sequences(M: List[Label], N: List[Label], k: int, seed_val: int | None = None) -> List[List[Label]]:
    """
    Genera k sucesiones válidas al azar:
    - (n-1) símbolos tomados de M
    - (m-1) símbolos tomados de N
    y luego se mezcla.
    """
    if seed_val is not None:
        rng = random.Random(seed_val)
        choice = rng.choice
        shuffle = rng.shuffle
    else:
        choice = random.choice
        shuffle = random.shuffle

    m = len(M)
    n = len(N)

    sequences: List[List[Label]] = []
    seen = set()

    tries = 0
    max_tries = max(100, 50 * k)

    while len(sequences) < k and tries < max_tries:
        tries += 1
        seq = [choice(M) for _ in range(n - 1)] + [choice(N) for _ in range(m - 1)]
        shuffle(seq)
        t = tuple(seq)
        if t not in seen:
            seen.add(t)
            sequences.append(seq)

    return sequences


# -----------------------------
# Dibujo (más pequeño y controlado)
# -----------------------------

def draw_bipartite_tree(
    G: nx.Graph,
    M: List[Label],
    N: List[Label],
    highlight_node: Label | None = None,
    highlight_edge: Edge | None = None,
    mode: str = "small",
):
    """
    mode:
      - "small": para fijar a la derecha sin ocupar pantalla
      - "medium": para ejemplos
    """
    pos = {}
    for i, a in enumerate(M):
        pos[a] = (0, i)
    for j, b in enumerate(N):
        pos[b] = (1, j)

    if mode == "small":
        fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=170)
        node_size = 280
        font_size = 7
        width = 1.1
    else:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=155)
        node_size = 430
        font_size = 8
        width = 1.2

    node_colors = ["lightcoral" if n in M else "lightblue" for n in G.nodes()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_size,
        font_size=font_size,
        width=width,
        ax=ax,
    )

    if highlight_edge is not None:
        u, v = highlight_edge
        if G.has_edge(u, v):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3.0, ax=ax)

    if highlight_node is not None and highlight_node in G.nodes():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[highlight_node],
            node_color="yellow",
            node_size=int(node_size * 1.35),
            edgecolors="black",
            linewidths=1.0,
            ax=ax,
        )

    ax.set_axis_off()
    fig.tight_layout()
    return fig


# -----------------------------
# Configuración Streamlit
# -----------------------------

st.set_page_config(page_title="K(m,n): Árboles Generadores", layout="wide")
st.title("🌳 Árboles generadores en el grafo bipartito completo K(m,n)")

st.markdown(
    """
Esta aplicación está inspirada en el artículo  
**"Spanning Trees of the Complete Bipartite Graph" de Hartsfield & Werth**.

Aquí puedes:
1. Ver cuántos árboles generadores tiene \(K_{m,n}\).
2. Ver ejemplos de árboles (con su sucesión asociada).
3. Ver animación **Árbol → Sucesión**.
4. Ver animación **Sucesión → Árbol** (con sucesiones válidas seleccionables).
"""
)

# Sidebar
st.sidebar.header("Parámetros de K(m,n)")
m = st.sidebar.slider("m (vértices en M)", min_value=2, max_value=6, value=3)
n = st.sidebar.slider("n (vértices en N)", min_value=2, max_value=6, value=3)

seed = st.sidebar.number_input(
    "Semilla aleatoria (0 = sin semilla fija)",
    min_value=0,
    value=0,
    step=1,
)
if seed != 0:
    random.seed(seed)

speed = st.sidebar.slider(
    "Velocidad de animación (segundos por paso)",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1,
)

M, N = build_complete_bipartite(m, n)
total_trees = count_spanning_trees(m, n)

st.sidebar.markdown(
    f"**Conjuntos:**  \n"
    f"M = {{ {', '.join(M)} }}  \n"
    f"N = {{ {', '.join(N)} }}"
)

st.subheader("Número total de árboles generadores de K(m,n)")
st.latex(r"\tau(K_{m,n}) = m^{n-1} \cdot n^{m-1}")
st.markdown(
    f"Para m = **{m}**, n = **{n}**:  \n"
    f"\\( \\tau(K_{{{m},{n}}}) = {m}^{{{n-1}}} \\cdot {n}^{{{m-1}}} = {total_trees} \\)"
)

# Tabs
tab_intro, tab_examples, tab_anim_forward, tab_anim_inverse, tab_theory = st.tabs(
    ["Introducción", "Ejemplos de árboles", "Animación Árbol → Sucesión", "Animación Sucesión → Árbol", "Teoría"]
)

# -----------------------------
# TAB: Introducción
# -----------------------------
with tab_intro:
    st.markdown("## 🧩 Introducción")
    st.markdown(
        """
El grafo **bipartito completo** \(K_{m,n}\) tiene:
- \\(M = \\{a_1, \\dots, a_m\\}\\) y \\(N = \\{b_1, \\dots, b_n\\}\\)
- Todas las aristas entre M y N

Un **árbol generador** es conexo, sin ciclos, y contiene todos los vértices.

Resultado clásico:
\\[
\\tau(K_{m,n}) = m^{n-1}\\, n^{m-1}
\\]
"""
    )

# -----------------------------
# TAB: Ejemplos (con ❓ sucesión asociada)
# -----------------------------
with tab_examples:
    st.markdown("## 🌲 Ejemplos de árboles generadores")

    max_show_all = 20
    num_samples = min(10, total_trees)

    if total_trees <= max_show_all:
        st.markdown(
            f"Como \\(\\tau(K_{{{m},{n}}}) = {total_trees} \\le {max_show_all}\\), "
            "intentaremos mostrar **todos** (por muestreo aleatorio; puede no salir el 100%)."
        )
        target = int(total_trees)
    else:
        st.markdown(
            f"El número total de árboles es **{total_trees}**, muy grande para verlos todos.  \n"
            f"Mostramos **{num_samples}** ejemplos aleatorios."
        )
        target = int(num_samples)

    sampled_edge_sets = set()
    sampled_trees: List[List[Edge]] = []
    max_tries = 1000 * max(1, target)
    tries = 0

    while len(sampled_trees) < target and tries < max_tries:
        tries += 1
        edges = random_spanning_tree_bipartite(M, N)
        normalized = tuple(sorted(tuple(sorted(e)) for e in edges))
        if normalized not in sampled_edge_sets:
            sampled_edge_sets.add(normalized)
            sampled_trees.append(edges)

    cols = st.columns(2)
    for idx, edges in enumerate(sampled_trees, start=1):
        G_tree = build_tree_graph(M, N, edges)
        fig = draw_bipartite_tree(G_tree, M, N, mode="medium")

        steps, seq = tree_to_sequence_steps(M, N, edges)

        col = cols[(idx - 1) % 2]
        with col:
            header = st.columns([0.78, 0.22])
            with header[0]:
                st.markdown(f"**Árbol {idx}**")
            with header[1]:
                with st.popover("❓"):
                    st.markdown("**Sucesión asociada**")
                    st.caption(f"Longitud: {len(seq)} (= m+n-2 = {m+n-2})")
                    st.code(str(seq))

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

# -----------------------------
# TAB: Animación Árbol → Sucesión (grafo pequeño a la derecha)
# -----------------------------
with tab_anim_forward:
    st.markdown("## 🔁 Animación: Árbol → Sucesión")

    left, right = st.columns([1.25, 0.75])

    with left:
        start_btn = st.button("🎬 Generar árbol y ver animación Árbol → Sucesión", key="btn_forward")
        placeholder_text = st.empty()
        progress_bar = st.progress(0.0)

    with right:
        st.markdown("### Árbol")
        placeholder_plot = st.empty()

    if start_btn:
        sim_edges = random_spanning_tree_bipartite(M, N)
        steps, seq = tree_to_sequence_steps(M, N, sim_edges)

        st.session_state.last_seq = seq
        st.session_state.last_M = list(M)
        st.session_state.last_N = list(N)
        st.session_state.last_edges = list(sim_edges)

        G_work = build_tree_graph(M, N, sim_edges)
        total_steps = len(steps)

        for i, step in enumerate(steps, start=1):
            removed = step["removed"]
            neighbor = step["neighbor"]

            if removed in G_work.nodes:
                G_work.remove_node(removed)

            fig = draw_bipartite_tree(G_work, M, N, highlight_node=neighbor, mode="small")
            placeholder_plot.pyplot(fig, use_container_width=False)
            plt.close(fig)

            placeholder_text.markdown(
                f"""
### Paso {i}/{total_steps}
- **Nodo eliminado:** `{removed}` (lado **{step['chosen_side']}**)
- **Vecino anotado en la sucesión:** `{neighbor}`
- **Sucesión parcial:** `{step['sequence']}`
"""
            )

            progress_bar.progress(i / total_steps)
            time.sleep(speed)

        progress_bar.progress(1.0)
        st.success(f"✅ Sucesión completa generada: `{seq}`")

# -----------------------------
# TAB: Animación Sucesión → Árbol (selector de sucesiones válidas + explicación)
# -----------------------------
with tab_anim_inverse:
    st.markdown("## 🔁 Animación: Sucesión → Árbol")

    left, right = st.columns([1.25, 0.75])

    # --------- derecha: selector + botón animación + árbol pequeño
    with right:
        st.markdown("### Selecciona una sucesión válida")

        # "semilla" para generar lista de sucesiones (independiente de la global si quieres)
        seq_seed = st.number_input(
            "Semilla para generar sucesiones (0 = aleatorio cada vez)",
            min_value=0,
            value=0,
            step=1,
            key="seq_seed",
        )
        k_options = st.slider("Cantidad de sucesiones a mostrar", 3, 12, 6, key="k_options")

        if st.button("🔄 Generar lista de sucesiones válidas", key="gen_seq_list"):
            seed_val = None if seq_seed == 0 else int(seq_seed)
            st.session_state.seq_options = generate_valid_sequences(M, N, k=int(k_options), seed_val=seed_val)
            st.session_state.selected_seq_idx = 0

        # generar por defecto si no existe
        if "seq_options" not in st.session_state:
            st.session_state.seq_options = generate_valid_sequences(M, N, k=int(k_options), seed_val=None)
            st.session_state.selected_seq_idx = 0

        options = st.session_state.seq_options
        option_labels = [", ".join(s) for s in options]

        selected_label = st.selectbox(
            "Sucesiones válidas disponibles:",
            options=option_labels,
            index=st.session_state.get("selected_seq_idx", 0),
            key="seq_selectbox",
        )
        selected_seq = options[option_labels.index(selected_label)]

        start_inv = st.button("🎬 Empezar animación con esta sucesión", key="btn_inverse")

        st.markdown("### Árbol")
        placeholder_plot2 = st.empty()

    # --------- izquierda: instrucciones + explicación validez + pasos detallados
    with left:
        st.markdown("### Instrucciones (en tiempo real)")

        expected_len = m + n - 2
        st.markdown(
            f"""
**Para \(K_{{m,n}}\) con m={m}, n={n}:**
- **Longitud requerida:** \(m+n-2 = {expected_len}\)
- **Símbolos permitidos:** `a1..a{m}` y `b1..b{n}`
- **Tipo de sucesión válida (conteos exactos):**
  - \(n-1 = {n-1}\) símbolos tipo `a` (lado M)
  - \(m-1 = {m-1}\) símbolos tipo `b` (lado N)
"""
        )

        st.markdown(f"**Sucesión seleccionada:** `{selected_seq}`")

        ok, reasons = validate_sequence(selected_seq, M, N)

        if ok:
            st.success("✅ Esta sucesión es válida para reconstrucción en este K(m,n).")
        else:
            st.error("❌ Esta sucesión NO es válida (esto no debería pasar si la lista se generó bien).")

        if reasons:
            st.markdown("**¿Por qué es válida? (chequeos)**")
            st.write(f"- Longitud: {reasons['actual_len']} (se esperaba {reasons['expected_len']})")
            if reasons["allowed_ok"]:
                st.write("- Símbolos: ✅ todos son de M ∪ N")
            else:
                st.write(f"- Símbolos: ❌ inválidos {reasons['bad_symbols']}")
            st.write(
                f"- Conteos: M={reasons['count_M']} (se esperaba {reasons['expected_count_M']}), "
                f"N={reasons['count_N']} (se esperaba {reasons['expected_count_N']})"
            )

        st.markdown("---")

        placeholder_text2 = st.empty()
        progress_bar2 = st.progress(0.0)

    # --------- ejecutar animación
    if start_inv:
        if not ok:
            st.error("No se puede animar: la sucesión seleccionada no es válida.")
        else:
            try:
                steps_inv, full_edges_inv = sequence_to_tree_steps(M, N, selected_seq)
            except Exception as e:
                st.error(f"Ocurrió un error reconstruyendo el árbol: {e}")
            else:
                total_steps_inv = len(steps_inv)

                for i, step in enumerate(steps_inv, start=1):
                    edges_so_far = step["edges_so_far"]
                    new_edge = step["edge"]

                    G_partial = build_tree_graph(M, N, edges_so_far)

                    fig2 = draw_bipartite_tree(
                        G_partial,
                        M,
                        N,
                        highlight_edge=new_edge,
                        mode="small",
                    )
                    placeholder_plot2.pyplot(fig2, use_container_width=False)
                    plt.close(fig2)

                    placeholder_text2.markdown(
                        f"""
### Paso {i}/{total_steps_inv}
- **Arista agregada:** `{new_edge}`
- **Aristas actuales:** {len(edges_so_far)}
"""
                    )
                    progress_bar2.progress(i / total_steps_inv)
                    time.sleep(speed)

                progress_bar2.progress(1.0)
                st.success("✅ Reconstrucción completa del árbol a partir de la sucesión seleccionada.")

# -----------------------------
# TAB: Teoría
# -----------------------------
with tab_theory:
    st.markdown("## 📚 Resumen teórico")
    st.markdown(
        r"""
**Teorema (Hartsfield–Werth):**  
\[
\tau(K_{m,n}) = m^{n-1}\, n^{m-1}.
\]

La demostración usa una biyección entre árboles generadores y sucesiones de longitud \(m+n-2\)
con símbolos en \(M \cup N\).
"""
    )
