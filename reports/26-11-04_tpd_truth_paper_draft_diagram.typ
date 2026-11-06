// Setup diagram for the paper draft (26-11-04_tpd_truth_paper_draft.typ): where the decomposition
// replaces the weights, how information reaches the loss, target vs non-target training.
// Note: cetz.draw exports functions named `fill` and `stroke`; don't use those as parameter names
// in functions that import cetz.draw.
#import "@preview/cetz:0.3.4"
#import "template.typ": palette

#let _dec = rgb("#d6e4f2")   // decomposed weights
#let _orig = rgb("#eceef1")  // original weights
#let _loss = rgb("#fbe3c4")  // positions whose output enters the loss
#let _chg = rgb("#e9f0f8")   // original weights, but the input already carries W's changes
#let _chg_edge = (paint: rgb("#7f9cbb"), thickness: 0.5pt, dash: "dashed")
#let _opt_edge = (paint: palette.assumed, thickness: 0.8pt, dash: "dashed")
#let _w = 0.98               // column pitch
#let _bw = 0.86              // box width

#let _col(i) = i * _w
#let _box(i, y0, y1, bg, body, edge: 0.4pt + palette.muted) = {
  import cetz.draw: rect, content
  rect((_col(i) - _bw / 2, y0), (_col(i) + _bw / 2, y1), fill: bg, stroke: edge, radius: 1.5pt)
  content(((_col(i)), (y0 + y1) / 2), body)
}

// rows (y ranges)
#let _r_tok = (0, 0.5)
#let _r_low = (0.75, 1.2)
#let _r_dec = (1.45, 2.15)
#let _r_up = (2.4, 2.85)
#let _r_out = (3.1, 3.6)

#let _labels = ("token", "layers 0–10", [MLP `down_proj` \ layers 11–17], "layers 18–27", "next-token output")
#let _rows() = {
  import cetz.draw: content
  for (k, lab) in _labels.enumerate() {
    let (a, b) = (_r_tok, _r_low, _r_dec, _r_up, _r_out).at(k)
    content((-0.55, (a + b) / 2), anchor: "east", text(size: 7pt, fill: palette.muted, lab))
  }
}

#let _tok(s) = text(size: 6.5pt, raw(s))
#let _small(s) = text(size: 6.5pt, s)
#let _arrow(c) = (end: ">", fill: c, scale: 0.55)

#let target_panel = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let stmt = ("The", "city", "of", "Paris", "is", "in", "France", ".\\n")
  let quest = ("The", "amount", "…", "is", "␣")
  let toks = stmt + quest
  let ns = stmt.len()
  let last = toks.len() - 1
  _rows()
  for (i, t) in toks.enumerate() {
    let is_stmt = i < ns
    _box(i, .._r_tok, if is_stmt { white } else { _orig }, _tok(t))
    _box(i, .._r_low, _orig, _small[$W$])
    if is_stmt {
      _box(i, .._r_dec, _dec, _small[$W'$], edge: 0.6pt + palette.accent)
    } else {
      _box(i, .._r_dec, _chg, _small[$W$], edge: _chg_edge)
    }
    _box(i, .._r_up, _chg, _small[$W$], edge: _chg_edge)
    if i == last {
      _box(i, .._r_out, _loss, _small[KL], edge: 0.8pt + palette.assumed)
    } else {
      _box(i, .._r_out, white, text(size: 6.5pt, fill: palette.rule)[—], edge: 0.3pt + palette.rule)
    }
  }
  // optional term on the decomposed matrices' outputs (recon, ortho) at the statement tokens
  let pad = 0.07
  rect((_col(0) - _bw / 2 - pad, _r_dec.at(0) - pad), (_col(ns - 1) + _bw / 2 + pad, _r_dec.at(1) + pad),
    stroke: _opt_edge, radius: 2.5pt)

  // brackets under the tokens
  let yb = -0.22
  line((_col(0) - _bw / 2, yb), (_col(ns - 1) + _bw / 2, yb), stroke: 0.6pt + palette.accent)
  content(((_col(0) + _col(ns - 1)) / 2, yb - 0.27), text(size: 6.5pt, fill: palette.accent)[statement: components replace `down_proj` here])
  line((_col(ns) - _bw / 2, yb), (_col(last) + _bw / 2, yb), stroke: 0.6pt + palette.muted)
  content(((_col(ns) + _col(last)) / 2, yb - 0.27), text(size: 6.5pt, fill: palette.muted)[question: original weights])
})

#let nontarget_panel = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let toks = ("The", "results", "of", "the", "study", "were", "publ", "ished", "in", "the", "journal", "Nature", "in")
  let last = toks.len() - 1
  _rows()
  for (i, t) in toks.enumerate() {
    _box(i, .._r_tok, white, _tok(t))
    _box(i, .._r_low, _orig, _small[$W$])
    _box(i, .._r_dec, _dec, _small[$W'$], edge: 0.6pt + palette.accent)
    _box(i, .._r_up, _chg, _small[$W$], edge: _chg_edge)
    _box(i, .._r_out, _loss, _small[KL], edge: 0.8pt + palette.assumed)
  }
  let yb = -0.22
  line((_col(0) - _bw / 2, yb), (_col(last) + _bw / 2, yb), stroke: 0.6pt + palette.accent)
  content(((_col(0) + _col(last)) / 2, yb - 0.27), text(size: 6.5pt, fill: palette.accent)[Pile window (64 tokens, 13 shown): components replace `down_proj` at every position])
  let ly = _r_out.at(1) + 0.35
})


// What the extra terms ask of one matrix's output at one statement token: y = W x splits
// exactly into C (all components at full strength) and D = Delta x, so C + D = y always.
#let _c_col = palette.accent
#let _d_col = rgb("#9a3d2c")
#let _y_vec = (2.4, 0.6)

#let _vec(a, b, col, lab, side: "west", w: 0.9pt) = {
  import cetz.draw: line, content
  line(a, b, stroke: w + col, mark: _arrow(col))
  let m = ((a.at(0) + b.at(0)) / 2, (a.at(1) + b.at(1)) / 2)
  content(m, anchor: side, padding: 0.08, text(size: 8pt, fill: col, lab))
}

#let _vec_panel(c, extra: none, sides: ("north-west", "south-east", "south-west")) = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // fixed frame so the three panels share one scale and baseline
  rect((-0.4, -0.5), (3.9, 2.9), stroke: none)
  if extra != none { extra }
  let o = (0, 0)
  _vec(o, _y_vec, black, $y$, side: sides.at(0))
  _vec(o, c, _c_col, $C$, side: sides.at(1))
  _vec(c, _y_vec, _d_col, $D$, side: sides.at(2))
  circle(o, radius: 0.04, fill: black, stroke: none)
})

#let restriction_panels = {
  let (yx, yy) = _y_vec
  // plain: C anywhere, D = y - C cancels the excess
  let c1 = (3.6, 2.6)
  // recon: C close to y, D small
  let c2 = (2.1, 0.18)
  // ortho: C on the circle whose diameter is the segment from 0 to y
  let (mx, my) = (yx / 2, yy / 2)
  let r = calc.sqrt(yx * yx + yy * yy) / 2
  let th = 112deg
  let c3 = (mx + r * calc.cos(th), my + r * calc.sin(th))
  let circ = {
    import cetz.draw: *
    // right-angle mark at C's tip, between the directions back to 0 and along D
    let (cx, cy) = c3
    let nc = calc.sqrt(cx * cx + cy * cy)
    let (dx, dy) = (yx - cx, yy - cy)
    let nd = calc.sqrt(dx * dx + dy * dy)
    let s = 0.17
    let u = (-cx / nc * s, -cy / nc * s)
    let v = (dx / nd * s, dy / nd * s)
    line((cx + u.at(0), cy + u.at(1)), (cx + u.at(0) + v.at(0), cy + u.at(1) + v.at(1)), (cx + v.at(0), cy + v.at(1)),
      stroke: 0.5pt + palette.muted)
  }
  let cap(head, body) = align(center, text(size: 7.5pt)[*#head* \ #text(fill: palette.muted, body)])
  grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 0.8em,
    row-gutter: 0.4em,
    align(center, _vec_panel(c1)), align(center, _vec_panel(c2, sides: ("south-east", "north-west", "west"))), align(center, _vec_panel(c3, extra: circ)),
    cap[plain tPD][no condition],
    cap[recon][$C approx y$ and $D approx 0$],
    cap[ortho][$chevron.l C, D chevron.r = 0$],
  )
}
