"""Figure helpers for the ENG2440 Assignment 3 notebook.

The general house style (fonts, colors, tick and frame rules, the automatic figure audit) lives in
`helpers/paper_plot_style.py`. This file adds what molecules need: a chemical structure drawing, the
matching node-edge view, a corner tag that carries the label and the predicted probability, and table
styling.

Drawing convention (stated in every molecule caption): structures are drawn black on white, so color is
never decoration. The only color on a molecule panel is its corner tag, which is red when the molecule is
labeled as penetrating the blood-brain barrier and petrol when it is not. Hydrogens are implicit, exactly
as they are in the graph the model sees.
"""
import io
import re

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from IPython.display import display
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from helpers.paper_plot_style import *            # noqa: F401,F403  (house style + audit)
from helpers.paper_plot_style import (COOL_GREY, COOL_INK, INK, LABEL_PT, MUTED, PETROL, RED,
                                      RULE, SLATE, TICK_PT, save_fig, text_color)

POSITIVE, NEGATIVE = RED, PETROL                  # the two models: GNN and fingerprint baseline
MONO = ["Courier New", "Courier", "DejaVu Sans Mono"]   # SMILES are monospaced; Courier pairs with Times
CLASS_NAME = {1: "penetrating", 0: "non-penetrating"}
SHORT_NAME = {1: "penetrating", 0: "non-pen."}


def tint(color, strength):
    """Mix `color` with white: strength 1 gives the full color, 0 gives white."""
    return tuple(1 - strength * (1 - c) for c in mpl.colors.to_rgb(color))


def readable_on(background):
    """White or dark ink, whichever contrasts more with `background` (WCAG relative luminance)."""
    def luminance(color):
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in mpl.colors.to_rgb(color)]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    bg = luminance(background)
    return "white" if 1.05 / (bg + 0.05) > (bg + 0.05) / (luminance(INK) + 0.05) else INK


def save_and_show(fig, name, width="full", rel=1.0):
    """Audit + save the figure (vector PDF and preview PNG under figures/), then show it."""
    save_fig(fig, name, width=width, rel=rel)
    display(fig)


# ------------------------------------------------------------------ molecules --
def draw_molecule(ax, smiles, size=(1760, 1100), bond_length=None):
    """Chemical structure, drawn black on white at `size` pixels and placed in `ax`.

    `size` is the bitmap RDKit renders, not the printed size. It has to be large because this drawing
    is the one rasterized thing in an otherwise vector figure: at the old 440 px it reached the page as
    a 100-dpi bitmap and printed soft beside the type. `bond_length` fixes the drawing scale in pixels
    per bond *of this bitmap*, so it must be rescaled whenever `size` changes so that several molecules in one figure are drawn at comparable size instead of each
    being auto-fitted to its own box, which shrank the large ones to a smudge.
    """
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.useBWAtomPalette()                    # every atom in ink; the corner tag carries the class
    options.bondLineWidth = 4                     # scaled with the bitmap, so the printed weight holds
    options.padding = 0.06
    if bond_length:
        options.fixedBondLength = bond_length
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    ax.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    return mol


def panel_label(ax, letter, text=None, pad=0.015):
    """House rule: every panel letter sits top-left, just outside the axes, left-aligned.

    Keeping one placement across all figures means a reader never has to hunt for the letter, and a
    label outside the frame can never land on the data.
    """
    label = f"({letter})" if text is None else f"({letter}) {text}"
    t = ax.text(0.0, 1.0 + pad, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=LABEL_PT, color=INK)
    t._direct_label = True          # outside the axes on purpose; the audit should not flag it
    return t


def class_tag(ax, label, probability=None, loc="upper left", y=0.985, va="top", prefix=""):
    """Tag naming the true class, and optionally the predicted probability beside it.

    The class is carried by the tag's *shape*, not its hue: penetrating is a filled dark badge,
    non-penetrating an outlined light one. Red and petrol are reserved for the two models elsewhere in
    the report, so a neutral badge here keeps a color from meaning two different things.
    """
    penetrating = int(label) == 1
    face, edge, ink = ((COOL_INK, COOL_INK, "white") if penetrating
                       else ("white", COOL_INK, COOL_INK))
    # Two lines when a probability is shown: the true class and the model's score are different things
    # and a single run-on pill blurred them, while a one-line version ran past the panel edge.
    text = f"{prefix}{CLASS_NAME[int(label)]}" if probability is None else \
        f"true: {CLASS_NAME[int(label)]}\npred. p = {probability:.2f}"
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=TICK_PT,
                   color=ink, zorder=5,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor=edge, linewidth=0.7))


def corner_tag(ax, text, color, loc="upper left"):
    """Pale-tinted tag in a given hue. Kept for anything that genuinely keys to a model color."""
    x, y, ha, va = (0.02, 0.98, "left", "top") if "left" in loc else (0.98, 0.98, "right", "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=TICK_PT,
            color=text_color(color, max_lum=0.32), zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=tint(color, 0.16),
                      edgecolor=tint(color, 0.45), linewidth=0.5))


# Atom tokens, longest first: a bracket atom, the two-letter organic-subset elements, a two-digit ring
# closure, then any single character. Wrapping only ever happens between whole tokens.
_SMILES_TOKEN = re.compile(r"\[[^\]]*\]|Br|Cl|%\d{2}|.")


def wrap_smiles(smiles, width=28, max_lines=4):
    """Break a SMILES string over up to `max_lines` lines so the whole string stays visible.

    Lines break between chemical tokens, never inside one: splitting `[C@@H]` across two lines would
    show a reader two fragments that are not what the string says.
    """
    lines, current = [], ""
    for token in _SMILES_TOKEN.findall(smiles):
        if current and len(current) + len(token) > width:
            lines.append(current)
            current = token
        else:
            current += token
    if current:
        lines.append(current)
    if len(lines) > max_lines:                        # only for strings no panel could hold
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1] + "…"
    return "\n".join(lines)


def molecule_panel(ax, smiles, label=None, probability=None, caption=None, loc="upper left",
                   bond_length=None):
    """One structure with its class tag; `caption` goes underneath in muted text.

    A tag drawn inside these axes has to share them with the structure. `molecule_grid` gives the tag a
    row of its own instead, which is the only way to guarantee it never touches an atom: widening the
    axes to make room shrinks the drawing, because an image keeps a fixed aspect ratio.
    """
    draw_molecule(ax, smiles, bond_length=bond_length)
    if label is not None:
        class_tag(ax, label, probability=probability, loc=loc)
    if caption:
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=TICK_PT - 1, color=MUTED, family=MONO, linespacing=1.25)
    return ax


def place_text(ax, text, prefer=(0.03, 0.97), **kw):
    """Put `text` in an empty part of `ax`, as close as possible to `prefer` (axes fractions, 0-1).

    Candidate positions on a 15 x 15 grid are tried from the nearest to the farthest; the first one whose
    text box stays inside the axes and touches no plotted line, no bar and no other text is kept.
    This uses the same geometry as the figure audit, so labels never need hand-tuned coordinates.
    """
    from helpers.paper_plot_style import _line_pts
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Avoid the curves of every axis sharing this screen rectangle, twins included: the audit now
    # checks the same way, so placement and audit can no longer disagree.
    neighbors = [a for a in fig.axes if a is ax or a.bbox.overlaps(ax.bbox)]
    points = [p for a in neighbors for line in a.lines for p in _line_pts(a, line)]
    boxes = [t.get_window_extent(renderer) for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes += [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    grid = [(x, y) for x in np.linspace(0.02, 0.98, 15) for y in np.linspace(0.03, 0.97, 15)]
    grid.sort(key=lambda xy: (xy[0] - prefer[0]) ** 2 + (xy[1] - prefer[1]) ** 2)
    kw = {"fontsize": TICK_PT, "color": INK, **kw}
    for x, y in grid:
        t = ax.text(x, y, text, transform=ax.transAxes, ha="left" if x < 0.5 else "right",
                    va="top" if y > 0.5 else "bottom", **kw)
        bb = t.get_window_extent(renderer)
        inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
        free = not any(bb.padded(2).contains(px, py) for px, py in points) and not any(bb.overlaps(b) for b in boxes)
        if inside and free:
            return t
        t.remove()
    print(f"place_text: no empty spot for {text[:30]!r}; placed at the preferred position")
    return ax.text(*prefer, text, transform=ax.transAxes, ha="left" if prefer[0] < 0.5 else "right",
                   va="top" if prefer[1] > 0.5 else "bottom", **kw)


def molecule_grid(fig, items, ncols=3, caption_ratio=0.40, caption_pt=None,
                  col_headers=None, width_ratios=None, tag_ratio=0.26, header_pt=None,
                  bond_length=None):
    """Grid of structures, each with a caption row of its own so a long SMILES always fits.

    `items` are dicts with keys `smiles`, `label` and optionally `probability` and `caption`; an entry
    of `None` leaves that cell empty, which is how a narrow spacer column separates groups.
    Giving the caption its own grid row (rather than hanging it under the image) is what keeps the
    bottom row of a multi-row figure from running off the page. `col_headers` names the columns above
    the first row, for grids whose columns mean something.
    """
    import math
    caption_pt = caption_pt or TICK_PT - 1
    nrows = math.ceil(len(items) / ncols)
    gs = fig.add_gridspec(3 * nrows, ncols,
                          height_ratios=[tag_ratio, 1.0, caption_ratio] * nrows,
                          width_ratios=width_ratios)
    axes = []
    for k, item in enumerate(items):
        row, col = divmod(k, ncols)
        if item is None:
            axes.append(None)
            continue
        tag_ax = fig.add_subplot(gs[3 * row, col])
        tag_ax.set_axis_off()
        if item.get("label") is not None:
            class_tag(tag_ax, item["label"], probability=item.get("probability"), y=0.5, va="center",
                      prefix=item.get("tag_prefix", ""))
        if col_headers and row == 0 and col < len(col_headers) and col_headers[col]:
            tag_ax.set_title(col_headers[col], fontsize=header_pt or TICK_PT, color=INK, pad=3)
        ax = fig.add_subplot(gs[3 * row + 1, col])
        molecule_panel(ax, item["smiles"], bond_length=bond_length)
        caption_ax = fig.add_subplot(gs[3 * row + 2, col])
        caption_ax.set_axis_off()
        cap = caption_ax.text(0.5, 1.0, item.get("caption", ""), transform=caption_ax.transAxes,
                              ha="center", va="top", fontsize=caption_pt, color=MUTED,
                              family=MONO, linespacing=1.25)
        cap._direct_label = True       # the caption row is a text-only axes; overflow is by design
        axes.append(ax)
    return axes


BOND_STYLE = {Chem.BondType.SINGLE: dict(linewidths=0.9, linestyles="-"),
              Chem.BondType.DOUBLE: dict(linewidths=2.2, linestyles="-"),
              Chem.BondType.TRIPLE: dict(linewidths=3.2, linestyles="-"),
              Chem.BondType.AROMATIC: dict(linewidths=1.6, linestyles=(0, (2.5, 1.5)))}
BOND_LEGEND = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
               "aromatic": Chem.BondType.AROMATIC}


def _bonds_by_style(mol):
    """Group a molecule's bonds by the line style their bond type is drawn with."""
    groups = {}
    for bond in mol.GetBonds():
        key = bond.GetBondType() if bond.GetBondType() in BOND_STYLE else Chem.BondType.SINGLE
        groups.setdefault(key, []).append(bond)
    return [(BOND_STYLE[k], v) for k, v in groups.items()]


def draw_graph(ax, smiles, node_size=95, label_atoms=True, bond_key=False):
    """The same molecule as nodes and edges, laid out on RDKit's 2D coordinates.

    Using the structure's own coordinates means an atom sits in the same place in both views, so the
    reader can follow one atom from the chemical drawing to its node.
    """
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])

    # Edges go in one LineCollection rather than one Line2D each: that is the idiomatic way to draw a
    # graph's edges, and it keeps the node labels (which sit on their nodes by design) off the line list.
    for style, bonds in _bonds_by_style(mol):
        segments = [[pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]] for b in bonds]
        ax.add_collection(LineCollection(segments, colors=RULE, zorder=1, capstyle="round", **style))
    # Two-letter symbols (Cl, Br) need a wider node than a single letter or the text touches the rim.
    sizes = [node_size * (1.5 if len(a.GetSymbol()) > 1 else 1.0) for a in mol.GetAtoms()]
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, facecolor="white", edgecolor=COOL_INK, linewidth=0.8,
               zorder=2)
    margin = 0.08 * max(float(np.ptp(pos, axis=0).max()), 1.0)
    ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
    ax.set_ylim(pos[:, 1].min() - margin * (2.4 if bond_key else 1.0), pos[:, 1].max() + margin)
    if bond_key:
        # A key inside the panel: line weight and dashes are the whole point of this view, and a reader
        # should not have to find the caption to learn what they mean.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for k, (name, kind) in enumerate(BOND_LEGEND.items()):
            x = x0 + (0.04 + k * 0.30) * (x1 - x0)
            y = y0 + 0.045 * (y1 - y0)
            ax.add_collection(LineCollection([[(x, y), (x + 0.11 * (x1 - x0), y)]], colors=RULE,
                                             **BOND_STYLE[kind]))
            ax.text(x + 0.125 * (x1 - x0), y, name, fontsize=TICK_PT - 1, color=MUTED,
                    ha="left", va="center")
    if label_atoms:
        for i, atom in enumerate(mol.GetAtoms()):
            ax.text(pos[i, 0], pos[i, 1], atom.GetSymbol(), ha="center", va="center",
                    fontsize=TICK_PT - 1, color=COOL_INK, zorder=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return mol, pos


# --------------------------------------------------------------------- tables --
def style_table(df, precision=3, caption=None):
    """Notebook table: no vertical rules, right-aligned numbers, muted header."""
    styled = (df.style
              .format(precision=precision)
              .set_table_styles([
                  {"selector": "th", "props": [("font-weight", "600"), ("color", INK),
                                               ("border-bottom", f"1px solid {RULE}"), ("text-align", "right")]},
                  {"selector": "td", "props": [("text-align", "right"), ("color", INK)]},
                  {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "13px")]}])
              .hide(axis="index"))
    if caption:
        styled = styled.set_caption(caption)
    return styled


# -------------------------------------------------------------------- curves --
def chance_line(ax, y, text, x=0.97):
    """Thin gray dashed reference line, labeled at its right end."""
    ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=0)
    ax.text(x, y, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=TICK_PT, color=MUTED)


def diagonal(ax, text="chance"):
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.7, ls="--", zorder=0)
    ax.text(0.62, 0.56, text, fontsize=TICK_PT, color=MUTED, rotation=45,
            rotation_mode="anchor", ha="center", va="bottom")


def confusion_panel(ax, counts, class_names=("non-pen.", "penetrating"), cmap_color=None,
                    show_ylabels=True):
    """2x2 confusion matrix as a tinted grid with counts and row percentages.

    Color maps to the row share alone, on the same 0-100% ramp in every panel, so a cell's shade can
    be compared across panels. Nothing else may be encoded in lightness here: letting the split also
    set the tint once made an 87% cell look paler than a 79% one.
    """
    matrix = np.asarray(counts, float)
    shares = matrix / matrix.sum(1, keepdims=True)
    base = cmap_color or PETROL
    for i in range(2):
        for j in range(2):
            face = tint(base, 0.06 + 0.94 * shares[i, j])
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=face, edgecolor="white", lw=1.2))
            # One rule for all four panels. readable_on() judged each cell separately, which put
            # near-black ink on the saturated red cells and white on the petrol ones, so the two
            # models looked like two different designs side by side.
            ink = "white" if shares[i, j] > 0.55 else INK
            ax.text(j + 0.5, 1.56 - i, f"{int(matrix[i, j])}", ha="center", va="center",
                    fontsize=TICK_PT + 1, color=ink)
            ax.text(j + 0.5, 1.22 - i, f"{shares[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=TICK_PT - 1, color=ink)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([1.5, 0.5])
    ax.set_xticklabels([f"pred.\n{n}" for n in class_names], fontsize=TICK_PT - 1)
    # Only the leftmost matrix of a row carries the actual-class labels; repeating them four times
    # cramps the row and tells the reader nothing new.
    ax.set_yticklabels([f"actual\n{n}" for n in class_names] if show_ylabels else ["", ""],
                       fontsize=TICK_PT - 1)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return ax
