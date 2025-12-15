# logic.py
"""
Lógica para árboles generadores de K(m,n)

Basado en:
- Hartsfield & Werth, "Spanning Trees of the Complete Bipartite Graph"

Funciones principales:
- build_complete_bipartite
- count_spanning_trees
- random_spanning_tree_bipartite
- tree_to_sequence_steps (Árbol → Sucesión)
- sequence_to_tree_steps (Sucesión → Árbol)
- build_tree_graph (para integración con NetworkX)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set
import random
import networkx as nx

Label = str
Edge = Tuple[Label, Label]


# -----------------------------
# Construcción básica
# -----------------------------

def build_complete_bipartite(m: int, n: int) -> Tuple[List[Label], List[Label]]:
    """
    Construye las etiquetas de los conjuntos M y N de K(m,n).

    M = {a1, ..., am}
    N = {b1, ..., bn}
    """
    M = [f"a{i}" for i in range(1, m + 1)]
    N = [f"b{j}" for j in range(1, n + 1)]
    return M, N


def count_spanning_trees(m: int, n: int) -> int:
    """
    Número de árboles generadores de K(m,n):
        τ(K_{m,n}) = m^{n-1} * n^{m-1}
    """
    return (m ** (n - 1)) * (n ** (m - 1))


def build_tree_graph(M: List[Label], N: List[Label], tree_edges: List[Edge]) -> nx.Graph:
    """
    Construye el grafo NetworkX del árbol a partir de M, N y la lista de aristas.
    """
    G = nx.Graph()
    G.add_nodes_from(M, bipartite=0)
    G.add_nodes_from(N, bipartite=1)
    G.add_edges_from(tree_edges)
    return G


# -----------------------------
# Generación de árbol aleatorio en K(m,n)
# -----------------------------

def random_spanning_tree_bipartite(M: List[Label], N: List[Label]) -> List[Edge]:
    """
    Genera un árbol generador aleatorio sobre el grafo bipartito completo K(m,n),
    respetando la bipartición (solo aristas M-N).

    Complejidad: O((m+n)^3) en el peor caso, pero para m,n pequeños (≤ 6–7) es más que suficiente.
    """
    M_set: Set[Label] = set(M)
    N_set: Set[Label] = set(N)
    nodes: List[Label] = M + N

    if not nodes:
        return []

    visited: Set[Label] = set()
    edges: List[Edge] = []

    # Elegimos nodo inicial al azar
    start: Label = random.choice(nodes)
    visited.add(start)

    total_nodes = len(nodes)

    while len(visited) < total_nodes:
        candidates: List[Edge] = []

        # Construimos todas las aristas posibles (u, v) con u en visitados y v no visitado
        for u in visited:
            if u in M_set:
                # u en M -> vecinos en N
                for v in N_set:
                    if v not in visited:
                        candidates.append((u, v))
            else:
                # u en N -> vecinos en M
                for v in M_set:
                    if v not in visited:
                        candidates.append((u, v))

        if not candidates:
            # Esto no debería ocurrir si K(m,n) es conexo,
            # pero por seguridad lanzamos excepción.
            raise RuntimeError("No se encontraron aristas candidatas; fallo en generación del árbol.")

        u, v = random.choice(candidates)
        edges.append((u, v))
        visited.add(v)

    # Verificación de seguridad
    G_check = build_tree_graph(M, N, edges)
    if not nx.is_tree(G_check) or len(edges) != (len(M) + len(N) - 1):
        raise RuntimeError("random_spanning_tree_bipartite no generó un árbol válido.")

    return edges


# -----------------------------
# Árbol → Sucesión
# -----------------------------

def numeric_index(label: Label) -> int:
    """
    Convierte 'a3' -> 3, 'b10' -> 10, útil para ordenar por índice.
    """
    return int(label[1:])


def tree_to_sequence_steps(
    M: List[Label],
    N: List[Label],
    edges: List[Edge],
) -> Tuple[List[Dict[str, Any]], List[Label]]:
    """
    Implementa el algoritmo "Árbol → Sucesión" (tipo Hartsfield–Werth).

    Dado un árbol T sobre K(m,n), produce:

    - steps: lista de pasos, cada uno con:
        {
          "removed": nodo hoja eliminado,
          "neighbor": vecino anotado en la sucesión,
          "sequence": sucesión parcial en ese momento,
          "remaining_edges": aristas restantes del grafo,
          "chosen_side": 'M' o 'N'
        }

    - final_sequence: la sucesión completa (lista de etiquetas)

    Proceso:
        Mientras el grafo tenga más de 2 nodos:
            1. Buscar hojas en N; si hay, tomar la de menor índice.
            2. Si no hay hojas en N, buscar en M y tomar la de menor índice.
            3. Anotar su vecino en la sucesión.
            4. Eliminar la hoja.
    """
    G = build_tree_graph(M, N, edges)

    if not nx.is_tree(G):
        raise ValueError("El grafo pasado a tree_to_sequence_steps no es un árbol.")

    seq: List[Label] = []
    steps: List[Dict[str, Any]] = []

    current_M: Set[Label] = set(M)
    current_N: Set[Label] = set(N)

    while len(G.nodes) > 2:
        leaves_N = [v for v in current_N if v in G.nodes and G.degree[v] == 1]
        chosen_side: str
        chosen_leaf: Label

        if leaves_N:
            leaves_N.sort(key=numeric_index)
            chosen_leaf = leaves_N[0]
            chosen_side = "N"
        else:
            leaves_M = [v for v in current_M if v in G.nodes and G.degree[v] == 1]
            if not leaves_M:
                raise RuntimeError("No se encontraron hojas; el grafo dejó de comportarse como árbol.")
            leaves_M.sort(key=numeric_index)
            chosen_leaf = leaves_M[0]
            chosen_side = "M"

        neighbor = list(G.neighbors(chosen_leaf))[0]
        seq.append(neighbor)

        remaining_edges = list(G.edges)
        steps.append(
            {
                "removed": chosen_leaf,
                "neighbor": neighbor,
                "sequence": list(seq),
                "remaining_edges": remaining_edges,
                "chosen_side": chosen_side,
            }
        )

        G.remove_node(chosen_leaf)
        if chosen_side == "M":
            current_M.remove(chosen_leaf)
        else:
            current_N.remove(chosen_leaf)

    expected_len = len(M) + len(N) - 2
    if len(seq) != expected_len:
        raise RuntimeError(
            f"La sucesión tiene longitud {len(seq)}, pero se esperaba {expected_len}."
        )

    return steps, seq


# -----------------------------
# Sucesión → Árbol
# -----------------------------

def sequence_to_tree_steps(
    M: List[Label],
    N: List[Label],
    sequence: List[Label],
) -> Tuple[List[Dict[str, Any]], List[Edge]]:
    """
    Implementa el algoritmo inverso "Sucesión → Árbol".

    Dada una sucesión S (lista de etiquetas en M ∪ N), reconstruye paso a paso
    el árbol asociado.

    Devuelve:
    - steps: lista de pasos, cada uno con:
        {
            "edge": arista agregada en este paso,
            "edges_so_far": lista de aristas hasta este paso
        }
    - final_edges: lista con todas las aristas del árbol reconstruido.
    """
    M_set: Set[Label] = set(M)
    N_set: Set[Label] = set(N)

    remaining_M: Set[Label] = set(M)
    remaining_N: Set[Label] = set(N)

    edges_so_far: List[Edge] = []
    steps: List[Dict[str, Any]] = []

    S = list(sequence)  # copia de trabajo

    while S:
        first = S[0]

        if first in M_set:
            # primer elemento es ai => conectamos con bj
            candidates = [b for b in remaining_N if b not in S]
            if not candidates:
                raise RuntimeError("No hay candidato adecuado en N para la reconstrucción.")
            candidates.sort(key=numeric_index)
            v = candidates[0]   # bj
            u = first           # ai
            remaining_N.remove(v)
            S = S[1:]
        elif first in N_set:
            # primer elemento es bj => conectamos con ai
            candidates = [a for a in remaining_M if a not in S]
            if not candidates:
                raise RuntimeError("No hay candidato adecuado en M para la reconstrucción.")
            candidates.sort(key=numeric_index)
            u = candidates[0]   # ai
            v = first           # bj
            remaining_M.remove(u)
            S = S[1:]
        else:
            raise ValueError(f"Etiqueta {first} no pertenece a M ∪ N.")

        edge = (u, v)
        edges_so_far.append(edge)
        steps.append(
            {
                "edge": edge,
                "edges_so_far": list(edges_so_far),
            }
        )

    # Última arista entre el único de M y el único de N que quedan
    if len(remaining_M) != 1 or len(remaining_N) != 1:
        raise RuntimeError("No queda exactamente un vértice en M y uno en N al final.")

    u_last = next(iter(remaining_M))
    v_last = next(iter(remaining_N))
    edge_last = (u_last, v_last)
    edges_so_far.append(edge_last)
    steps.append(
        {
            "edge": edge_last,
            "edges_so_far": list(edges_so_far),
        }
    )

    return steps, edges_so_far
