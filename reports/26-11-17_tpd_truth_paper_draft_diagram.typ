// Conceptual figures for the paper draft 26-11-17_tpd_truth_paper_draft.typ:
// - tpd_figure (introduction, generic tPD): one decomposed matrix during training (masked rank-1
//   components + remainder Delta, CI network);
// - setup_figure (setting, this paper): (a) the target stream (statement + probe, verdict KL),
//   (b) the non-target stream (Pile, next-token KL everywhere).
// Replaces the token-by-layer grid of 26-11-04_tpd_truth_paper_draft_diagram.typ.
// Note: cetz.draw exports functions named `fill` and `stroke`; don't use those as parameter names.
#import "@preview/cetz:0.3.4"
#import "template.typ": palette

#let _c_col = palette.accent          // components (same colour as C in the restriction figure)
#let _d_col = rgb("#9a3d2c")          // Delta (same colour as D in the restriction figure)
#let _c_bg = rgb("#d6e4f2")
#let _d_bg = rgb("#f3dcd6")
#let _orig = rgb("#eceef1")
#let _orig_bar = rgb("#9aa3ad")
#let _loss = palette.assumed
#let _opt_edge = (paint: palette.assumed, thickness: 0.8pt, dash: "dashed")

#let _t(s, size: 7pt, fill: black) = text(size: size, fill: fill, s)
#let _arrow(c) = (end: ">", fill: c, scale: 0.5)

// ---------------------------------------------------------------- (a) one decomposed matrix

// rank-1 component U_c V_c^T as a column times a row
#let _rank1(x, y, s) = {
  import cetz.draw: rect
  let th = 0.13
  rect((x, y), (x + s, y + s), fill: white, stroke: 0.4pt + _c_col.lighten(40%), radius: 1pt)
  rect((x, y), (x + th, y + s), fill: _c_bg.darken(8%), stroke: 0.5pt + _c_col)            // U_c
  rect((x + th + 0.05, y + s - th), (x + s, y + s), fill: _c_bg.darken(8%), stroke: 0.5pt + _c_col) // V_c^T
}

#let _full(x, y, s) = {
  import cetz.draw: rect, line
  rect((x, y), (x + s, y + s), fill: _d_bg, stroke: 0.6pt + _d_col, radius: 1pt)
}

#let decomposition_panel = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let s = 0.95
  let y = 0
  content((0.45, y + s / 2), text(size: 10pt)[$W' =$])
  // terms: (mask label, kind, x of icon, label under)
  let terms = (
    ($m_1$, "c", 1.6, $U_1 V_1^top$),
    ($m_2$, "c", 3.55, $U_2 V_2^top$),
    (none, "dots", 5.1, none),
    ($m_N$, "c", 6.35, $U_N V_N^top$),
    ($m_Delta$, "d", 8.35, [$Delta$ (full rank)]),
  )
  for (k, (m, kind, x, lab)) in terms.enumerate() {
    if kind == "dots" {
      content((x, y + s / 2), text(size: 10pt)[$dots.c$])
    } else {
      content((x - 0.08, y + s / 2), anchor: "east", text(size: 9pt, fill: if kind == "d" { _d_col } else { _c_col }, m))
      if kind == "c" { _rank1(x, y, s) } else { _full(x, y, s) }
      content((x + s / 2, y - 0.3), text(size: 7.5pt, lab))
    }
    if k < terms.len() - 1 {
      let xn = terms.at(k + 1).at(2)
      let xp = if kind == "dots" { x + 0.2 } else { x + s }
      let xm = if terms.at(k + 1).at(1) == "dots" { xn - 0.2 } else { xn - 0.62 }
      content(((xp + xm) / 2, y + s / 2), text(size: 10pt)[$+$])
    }
  }
  // braces
  let yb = y + s + 0.18
  line((1.6, yb), (1.6, yb + 0.08), (6.35 + s, yb + 0.08), (6.35 + s, yb), stroke: 0.5pt + _c_col)
  content(((1.6 + 6.35 + s) / 2, yb + 0.32), _t(fill: _c_col)[$N$ rank-1 *components*])
  line((8.35, yb), (8.35, yb + 0.08), (8.35 + s, yb + 0.08), (8.35 + s, yb), stroke: 0.5pt + _d_col)
  content((8.35 + s / 2, yb + 0.32), _t(fill: _d_col)[*remainder*])
  content((5.0, y - 0.75), _t(fill: palette.muted)[with every mask at 1, $W' = W$ exactly])

  // CI network on the right
  let x0 = 9.95
  content((x0, y + s / 2), anchor: "west", align(left, _t[token's \ input $x$]))
  line((x0 + 0.8, y + s / 2), (x0 + 1.15, y + s / 2), stroke: 0.5pt + black, mark: _arrow(black))
  rect((x0 + 1.2, y + 0.15), (x0 + 2.6, y + s - 0.15), fill: _orig, stroke: 0.5pt + palette.muted, radius: 2pt)
  content((x0 + 1.9, y + s / 2), _t[CI network])
  content((x0 + 1.9, y + s + 0.05), text(size: 6pt, fill: palette.muted)[shared by all decomposed matrices])
  line((x0 + 2.65, y + s / 2), (x0 + 3.0, y + s / 2), stroke: 0.5pt + black, mark: _arrow(black))
  // CI values: few high, most near 0
  let cis = (0.95, 0.04, 0.0, 0.8, 0.02, 0.0, 0.05, 0.6, 0.0, 0.03)
  let bx = x0 + 3.15
  let bw = 0.25
  line((bx - 0.03, y + 0.05), (bx + cis.len() * bw + 0.03, y + 0.05), stroke: 0.4pt + palette.muted)
  for (i, v) in cis.enumerate() {
    if v > 0 {
      rect((bx + i * bw + 0.03, y + 0.05), (bx + (i + 1) * bw - 0.03, y + 0.05 + v * (s - 0.05)), fill: _c_col, stroke: none)
    }
  }
  content((bx + cis.len() * bw / 2, y - 0.25), _t(size: 7pt)[$"CI"_c in [0, 1]$, one per component])
  content((x0 + 3.2, y - 0.75), _t(fill: palette.muted)[a penalty on all CI values keeps few components active])
})

// ---------------------------------------------------------------- (b), (c) the two data streams

#let _pw = 7.7          // panel width
#let _rows = (          // (y0, y1) of: text, layers 0-10, MLP 11-17, layers 18-27
  (0, 0.8), (1.05, 1.45), (1.7, 2.35), (2.85, 3.25),
)
#let _row_box(x0, x1, r, bg, edge, body) = {
  import cetz.draw: rect, content
  let (a, b) = _rows.at(r)
  rect((x0, a), (x1, b), fill: bg, stroke: edge, radius: 2pt)
  content(((x0 + x1) / 2, (a + b) / 2), body)
}
#let _ups(xs) = {
  import cetz.draw: line
  for r in range(3) {
    let a = _rows.at(r).at(1)
    let b = _rows.at(r + 1).at(0)
    for x in xs { line((x, a + 0.02), (x, b - 0.02), stroke: 0.4pt + palette.muted, mark: _arrow(palette.muted)) }
  }
}

// a pair of bars per outcome: original (grey) and with W' (blue)
#let _bars(x, y, pairs, h: 0.85, bw: 0.17, labels: none) = {
  import cetz.draw: rect, line, content
  for (i, (o, d)) in pairs.enumerate() {
    let xi = x + i * (2 * bw + 0.14)
    rect((xi, y), (xi + bw, y + o * h), fill: _orig_bar, stroke: none)
    rect((xi + bw, y), (xi + 2 * bw, y + d * h), fill: _c_col, stroke: none)
    if labels != none { content((xi + bw, y - 0.17), text(size: 6.5pt, raw(labels.at(i)))) }
  }
  line((x - 0.05, y), (x + pairs.len() * (2 * bw + 0.14) - 0.09, y), stroke: 0.4pt + palette.muted)
}

#let target_stream = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let xs = 3.75                     // end of the statement
  let xp = xs + 0.1                 // start of the probe
  _row_box(0, xs, 0, white, 0.6pt + _c_col,
    _t(size: 7.5pt)[_The city of Paris is in France._])
  _row_box(xp, _pw, 0, _orig, 0.4pt + palette.muted,
    align(center, _t(size: 7pt)[_The amount of truth in \ this statement is_ #raw("␣")]))
  _row_box(0, _pw, 1, _orig, 0.4pt + palette.muted, _t(fill: palette.muted)[layers 0–10])
  _row_box(0, xs, 2, _c_bg, 0.7pt + _c_col, _t[MLP output $W'$, layers 11–17])
  _row_box(xp, _pw, 2, _orig, 0.4pt + palette.muted, _t(fill: palette.muted)[original $W$])
  _row_box(0, _pw, 3, _orig, 0.4pt + palette.muted, _t(fill: palette.muted)[layers 18–27])
  // optional extra term (recon / ortho) on the components' write at the statement tokens
  let (a, b) = _rows.at(2)
  rect((-0.07, a - 0.07), (xs + 0.07, b + 0.07), stroke: _opt_edge, radius: 3pt)
  _ups((0.9, 2.85, 6.1, _pw - 0.35))
  // attention: the probe reads the statement
  bezier((xs - 0.4, b + 0.03), (xp + 1.0, _rows.at(3).at(0) - 0.03), (xs + 0.1, b + 0.42),
    stroke: 0.6pt + _c_col, mark: _arrow(_c_col))
  content((xp + 1.1, (b + _rows.at(3).at(0)) / 2), anchor: "west", text(size: 6.5pt, fill: _c_col)[attention])
  // output at the last probe token
  let yo = _rows.at(3).at(1) + 0.25
  line((_pw - 0.35, _rows.at(3).at(1) + 0.02), (_pw - 0.35, yo - 0.02), stroke: 0.4pt + palette.muted, mark: _arrow(palette.muted))
  _bars(_pw - 1.05, yo + 0.22, ((0.88, 0.85), (0.1, 0.13)), labels: ("1", "0"))
  content((_pw - 1.2, yo + 0.6), anchor: "east", align(right, _t(fill: _loss)[*KL* at the answer \ #text(fill: black)[(the verdict)]]))
})

#let nontarget_stream = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  _row_box(0, _pw, 0, white, 0.6pt + _c_col,
    _t(size: 7.5pt)[_… the results of the study were published in the journal Nature in …_])
  _row_box(0, _pw, 1, _orig, 0.4pt + palette.muted, _t(fill: palette.muted)[layers 0–10])
  _row_box(0, _pw, 2, _c_bg, 0.7pt + _c_col, _t[MLP output $W'$, layers 11–17, at every token])
  _row_box(0, _pw, 3, _orig, 0.4pt + palette.muted, _t(fill: palette.muted)[layers 18–27])
  let xs = (0.55, 1.85, 3.15, 4.45, 5.75, 7.05)
  _ups(xs)
  let yo = _rows.at(3).at(1) + 0.25
  let dists = (((0.5, 0.45), (0.3, 0.33), (0.2, 0.2)), ((0.7, 0.66), (0.2, 0.22), (0.1, 0.1)),
    ((0.35, 0.4), (0.35, 0.3), (0.3, 0.3)), ((0.8, 0.75), (0.15, 0.2), (0.05, 0.05)),
    ((0.45, 0.5), (0.4, 0.35), (0.15, 0.15)), ((0.6, 0.6), (0.3, 0.28), (0.1, 0.12)))
  for (x, d) in xs.zip(dists) {
    line((x, _rows.at(3).at(1) + 0.02), (x, yo - 0.02), stroke: 0.4pt + palette.muted, mark: _arrow(palette.muted))
    _bars(x - 0.45, yo + 0.22, d, h: 0.75, bw: 0.1, )
  }
  content((_pw / 2, yo + 1.2), _t(fill: _loss)[*KL* of the next-token prediction at every token])
})

// ---------------------------------------------------------------- assembled figure

#let _head(s) = text(size: 8.5pt, weight: "bold", s)
#let _note(body) = block(width: 100%, inset: (x: 2pt), text(size: 7.5pt, body))

// Generic tPD (introduction): one decomposed matrix with its masks and the CI network.
#let tpd_figure = block(width: 100%, align(center, decomposition_panel))

// This paper's training setup (setting section): the target and the non-target stream.
#let setup_figure = block(width: 100%)[
  #set align(left)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 1.4em,
    row-gutter: 0.5em,
    _head[(a) Target data: a statement, then the truth probe],
    _head[(b) Non-target data: ordinary text (the Pile)],
    align(center, target_stream),
    align(center + bottom, nontarget_stream),
    _note[
      *Masks:* $m_c = "CI"_c + (1 - "CI"_c) u$, $u tilde U[0, 1]$; $m_Delta tilde U[0, 1]$. \
      *Pressure:* $Delta$ is scaled down at random, so the few components with high CI must produce the verdict. \
      #text(fill: _loss)[Dashed: recon and ortho add a term on what the components write here (@fig-restrictions).]
    ],
    _note[
      *Masks:* $m_c$ random in $["CI"_c, 1]$; $m_Delta = 1$. \
      *Pressure:* whatever the layers do on ordinary text is free if it sits in $Delta$ but costs CI if it sits in a component, so it moves to $Delta$.
    ],
  )
  #v(0.3em)
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1.2em,
    _t[#box(width: 0.6em, height: 0.6em, fill: _orig_bar) original model],
    _t[#box(width: 0.6em, height: 0.6em, fill: _c_col) model with $W'$],
    _t[#box(width: 0.6em, height: 0.6em, fill: _orig, stroke: 0.4pt + palette.muted) original weights],
  )
]
