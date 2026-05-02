"""
routing.py — Personalised Route Weighting

Component 2 of the system. Once the classifier identifies a driver's profile,
this module builds a weighted road graph and finds the best route for that
profile using Dijkstra's algorithm.

The core idea: each road type gets a different cost multiplier per profile.
Conservative drivers avoid highways and shortcuts; Spirited drivers love them.

  Conservative: surface ×0.7  highway ×2.5  shortcut ×1.5  → quiet residential path
  Normal:       all ×1.0                                     → balanced arterial path
  Spirited:     surface ×1.3  highway ×1.0  shortcut ×0.6  → highway + bypass route

The graph has 12 nodes and three structurally distinct paths from Home to OfficeE:
  Path A — residential arc      (Conservative's pick)
  Path B — arterial + shortcut  (Normal's pick)
  Path C — highway + bypass     (Spirited's pick)

Call build_graph() / get_routes() / plot_routes() to use this as a module.
Running the file directly runs a standalone demo.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ROUTE_FIGURE = os.path.join(FIGURES_DIR, "route_comparison.png")

# ---------------------------------------------------------------------------
# Profile constants
# ---------------------------------------------------------------------------
PROFILES = ("Conservative", "Normal", "Spirited")

PROFILE_COLOURS = {
    "Conservative": "#2196F3",   # blue
    "Normal":       "#FFC107",   # gold / amber
    "Spirited":     "#F44336",   # red
}

# Road-type cost multipliers per driver profile
WEIGHT_MULTIPLIERS = {
    "Conservative": {"surface": 0.7,  "highway": 2.5, "shortcut": 1.5},
    "Normal":       {"surface": 1.0,  "highway": 1.0, "shortcut": 1.0},
    "Spirited":     {"surface": 1.3,  "highway": 1.0, "shortcut": 0.6},
}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph() -> nx.DiGraph:
    """
    Build and return the 12-node directed city graph.

    Node groups:
      Residential (north arc): Home, ResStreet1, ResStreet2, ResJunction
      Arterial + shortcut:     ArtMain, ShortcutA
      Highway (south arc):     OnRamp, HwyMid, HwyExit, BypassMid
      Destination:             OfficeParking, OfficeE

    Weighted costs by profile (base = distance_km / speed_limit):
                        Conservative   Normal   Spirited
      Path A (residential): 0.1190     0.1700   0.2210  ← Conservative wins
      Path B (arterial):    0.1372     0.1275   0.1237  ← Normal wins
      Path C (highway):     0.2225     0.1318   0.1043  ← Spirited wins
    """
    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # Nodes with (x, y) positions for layout
    # ------------------------------------------------------------------
    positions = {
        # Residential northern arc (Path A)
        "Home":         (0.0, 2.0),
        "ResStreet1":   (1.2, 3.2),
        "ResStreet2":   (2.6, 3.8),
        "ResJunction":  (3.9, 3.0),
        # Arterial + shortcut middle corridor (Path B)
        "ArtMain":      (1.6, 1.2),
        "ShortcutA":    (3.6, 1.0),
        # Highway southern arc (Path C)
        "OnRamp":       (0.5, 0.2),
        "HwyMid":       (2.5, 0.0),
        "HwyExit":      (4.5, 0.3),
        "BypassMid":    (5.1, 1.6),
        # Destination cluster
        "OfficeParking": (5.0, 2.6),
        "OfficeE":       (5.6, 3.4),
    }
    for node, (x, y) in positions.items():
        G.add_node(node, pos=(x, y))

    # ------------------------------------------------------------------
    # Helper: add a bidirectional edge
    # ------------------------------------------------------------------
    def add_edge(u: str, v: str, dist_km: float,
                 speed_limit: int, road_type: str):
        base = dist_km / speed_limit
        attrs = dict(distance_km=dist_km, speed_limit=speed_limit,
                     road_type=road_type, base_cost=base)
        G.add_edge(u, v, **attrs)
        G.add_edge(v, u, **attrs)

    # ------------------------------------------------------------------
    # Path A  — residential arc (all surface)
    # base costs: 0.050 + 0.050 + 0.050 + 0.010 + 0.010 = 0.170
    # ------------------------------------------------------------------
    add_edge("Home",        "ResStreet1",   1.5, 30, "surface")
    add_edge("ResStreet1",  "ResStreet2",   1.5, 30, "surface")
    add_edge("ResStreet2",  "ResJunction",  1.0, 20, "surface")
    add_edge("ResJunction", "OfficeParking",0.3, 30, "surface")

    # ------------------------------------------------------------------
    # Path B  — arterial + urban shortcut
    # base costs: 0.050 + 0.060 + 0.0075 = 0.1175  (+0.010 shared = 0.1275)
    # ------------------------------------------------------------------
    add_edge("Home",      "ArtMain",      2.0, 40, "surface")   # base 0.050
    add_edge("ArtMain",   "ShortcutA",    1.8, 30, "shortcut")  # base 0.060
    add_edge("ShortcutA", "OfficeParking",0.3, 40, "surface")   # base 0.0075

    # ------------------------------------------------------------------
    # Path C  — highway spine + bypass shortcut
    # base costs: 0.005+0.005+0.0318+0.040+0.040 = 0.1218 (+0.010 shared = 0.1318)
    # ------------------------------------------------------------------
    add_edge("Home",      "OnRamp",  0.3, 60,  "surface")   # base 0.005
    add_edge("OnRamp",    "HwyMid",  0.5, 100, "highway")   # base 0.005
    add_edge("HwyMid",    "HwyExit", 3.5, 110, "highway")   # base 0.0318
    add_edge("HwyExit",   "BypassMid", 1.2, 30, "shortcut") # base 0.040
    add_edge("BypassMid", "OfficeParking", 1.2, 30, "shortcut") # base 0.040

    # ------------------------------------------------------------------
    # Shared terminal edge
    # ------------------------------------------------------------------
    add_edge("OfficeParking", "OfficeE", 0.3, 30, "surface")  # base 0.010

    # ------------------------------------------------------------------
    # Cross-connections  (add realism; verified not to create cheaper
    # routes than each profile's intended path)
    # ------------------------------------------------------------------
    # ResStreet1 ↔ ArtMain  (cut-through to arterial)
    add_edge("ResStreet1", "ArtMain",   1.0, 40, "surface")   # base 0.025

    # ResJunction ↔ HwyExit  (residential shortcut to highway exit area)
    add_edge("ResJunction", "HwyExit",  2.5, 30, "surface")   # base 0.083

    # ArtMain ↔ HwyExit  (arterial continues toward office area)
    add_edge("ArtMain", "HwyExit",      3.0, 30, "surface")   # base 0.100

    # BypassMid ↔ ShortcutA  (connecting both shortcut zones)
    add_edge("BypassMid", "ShortcutA",  0.8, 25, "shortcut")  # base 0.032

    return G


# ---------------------------------------------------------------------------
# Weight function
# ---------------------------------------------------------------------------
def edge_weight(_u: str, _v: str, edge_data: dict, profile: str) -> float:
    """
    Profile-adjusted edge cost for Dijkstra's algorithm.

    Base cost = distance_km / speed_limit  (travel-time proxy)
    Multiplied by the road-type factor for the given driver profile.
    """
    mult = WEIGHT_MULTIPLIERS[profile].get(edge_data["road_type"], 1.0)
    return edge_data["base_cost"] * mult


# ---------------------------------------------------------------------------
# Route finder
# ---------------------------------------------------------------------------
def get_routes(G: nx.DiGraph, origin: str, destination: str) -> dict:
    """
    Run profile-weighted Dijkstra from origin to destination for all three profiles.
    Returns a dict keyed by profile name; each value has path, edges, and total cost.
    """
    results = {}
    for profile in PROFILES:
        def _w(u, v, d, _p=profile):
            return edge_weight(u, v, d, _p)

        path = nx.dijkstra_path(G, origin, destination, weight=_w)
        cost = nx.dijkstra_path_length(G, origin, destination, weight=_w)
        edges = [
            (u, v, G[u][v]["road_type"], G[u][v]["distance_km"])
            for u, v in zip(path[:-1], path[1:])
        ]
        results[profile] = {"path": path, "edges": edges, "cost": cost}

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_routes(
    G: nx.DiGraph,
    routes: dict,
    origin: str,
    destination: str,
    save_path: str = ROUTE_FIGURE,
):
    """
    Draw the full graph with all three profile routes overlaid in colour.
    Conservative = blue, Normal = gold, Spirited = red.
    Saves the figure to *save_path*.
    """
    pos = nx.get_node_attributes(G, "pos")

    fig, ax = plt.subplots(figsize=(17, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F7FA")

    # --- Background edges, styled by road type -----------------------
    road_style = {
        "highway":  dict(edge_color="#90A4AE", width=2.5, style="solid"),
        "surface":  dict(edge_color="#B0BEC5", width=1.5, style="solid"),
        "shortcut": dict(edge_color="#A1887F", width=2.0, style="dashed"),
    }
    for rtype, style in road_style.items():
        subset = [(u, v) for u, v, d in G.edges(data=True)
                  if d["road_type"] == rtype]
        nx.draw_networkx_edges(
            G, pos, edgelist=subset, ax=ax,
            arrows=False, alpha=0.35, **style,
        )

    # --- Profile routes — straight arrows, draw Normal first ---------
    line_cfg = {
        "Conservative": dict(width=5, arrowsize=25),
        "Normal":        dict(width=4, arrowsize=22),
        "Spirited":      dict(width=5, arrowsize=25),
    }
    draw_order = ["Normal", "Conservative", "Spirited"]
    for profile in draw_order:
        path  = routes[profile]["path"]
        elist = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            edge_color=PROFILE_COLOURS[profile],
            alpha=0.90, arrows=True,
            **line_cfg[profile],
        )

    # --- Nodes --------------------------------------------------------
    node_colours, node_sizes, node_borders = [], [], []
    for node in G.nodes():
        if node == origin:
            node_colours.append("#66BB6A"); node_sizes.append(3000)
            node_borders.append("#2E7D32")
        elif node == destination:
            node_colours.append("#EF5350"); node_sizes.append(3000)
            node_borders.append("#B71C1C")
        else:
            node_colours.append("#ECEFF1"); node_sizes.append(3000)
            node_borders.append("#546E7A")

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colours, node_size=node_sizes,
        ax=ax, alpha=1.0, edgecolors=node_borders, linewidths=1.8,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=7.5, font_color="black",
        font_weight="bold", ax=ax,
    )

    # --- Edge labels (road-type + distance) only on route edges ------
    route_edge_set: set[tuple] = set()
    for r in routes.values():
        for u, v, *_ in r["edges"]:
            route_edge_set.add((u, v))

    abbr = {"highway": "HWY", "surface": "SRF", "shortcut": "SCT"}
    elabels = {
        (u, v): f"{abbr[d['road_type']]}  {d['distance_km']} km"
        for u, v, d in G.edges(data=True)
        if (u, v) in route_edge_set
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=elabels, font_size=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        ax=ax,
    )

    # --- Legend (placed below axes to avoid overlapping the graph) ---
    _route_km = {p: sum(e[3] for e in routes[p]["edges"]) for p in PROFILES}
    legend_handles = [
        mpatches.Patch(color=PROFILE_COLOURS["Conservative"],
                       label=f"Conservative — {_route_km['Conservative']:.1f} km  cost {routes['Conservative']['cost']:.4f}"),
        mpatches.Patch(color=PROFILE_COLOURS["Normal"],
                       label=f"Normal — {_route_km['Normal']:.1f} km  cost {routes['Normal']['cost']:.4f}"),
        mpatches.Patch(color=PROFILE_COLOURS["Spirited"],
                       label=f"Spirited — {_route_km['Spirited']:.1f} km  cost {routes['Spirited']['cost']:.4f}"),
        mpatches.Patch(color="#66BB6A", label=f"Origin ({origin})"),
        mpatches.Patch(color="#EF5350", label=f"Destination ({destination})"),
        mpatches.Patch(color="#90A4AE", label="Highway"),
        mpatches.Patch(color="#B0BEC5", label="Surface road"),
        mpatches.Patch(color="#A1887F", label="Shortcut (dashed)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=9, framealpha=0.93, edgecolor="#CCCCCC",
              handlelength=1.5)

    # --- Weight multiplier annotation (academic context) -------------
    weight_note = (
        "Profile cost multipliers  (base cost = distance ÷ speed limit)\n"
        "               Conservative    Normal    Spirited\n"
        "  Surface  :      ×0.7          ×1.0       ×1.3\n"
        "  Highway  :      ×2.5          ×1.0       ×1.0\n"
        "  Shortcut :      ×1.5          ×1.0       ×0.6"
    )
    ax.text(
        0.015, 0.985, weight_note, transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CCCCCC", alpha=0.92),
    )

    ax.set_title(
        "Personalised GPS Route Recommendations\n"
        f"Origin: {origin}  →  Destination: {destination}",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Route visualisation saved → {save_path}")


# ---------------------------------------------------------------------------
# Pretty-print helper (used by main.py)
# ---------------------------------------------------------------------------
def print_route(profile: str, route_data: dict):
    """Print a human-readable summary of one profile's route."""
    path     = route_data["path"]
    edges    = route_data["edges"]
    cost     = route_data["cost"]
    total_km = sum(e[3] for e in edges)
    print(f"\n  [{profile}]")
    print(f"    Path          : {' → '.join(path)}")
    print(f"    Total distance: {total_km:.1f} km")
    print(f"    Weighted cost : {cost:.4f}")
    print("    Edges:")
    for u, v, road_type, dist in edges:
        print(f"      {u:>13} → {v:<13}  {road_type:<9}  {dist:.1f} km")


# ---------------------------------------------------------------------------
# Module-level demo  (runs only when executed directly)
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("  Routing Demo — Personalised GPS Route Weighting")
    print("=" * 62)

    G           = build_graph()
    origin      = "Home"
    destination = "OfficeE"

    print(f"\n  Graph : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} directed edges")
    print(f"  Query : {origin}  →  {destination}\n")

    routes = get_routes(G, origin, destination)

    # Verify distinctness
    paths  = [tuple(routes[p]["path"]) for p in PROFILES]
    unique = len(set(paths))
    print(f"  Distinct routes: {unique} / {len(PROFILES)}\n")

    for profile in PROFILES:
        print_route(profile, routes[profile])

    print()
    plot_routes(G, routes, origin, destination)


if __name__ == "__main__":
    main()
