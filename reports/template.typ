// Shared layout for project reports in reports/.
//
// Reports are status documents with a shelf life, not papers: name them
// `YY-MM-DD_<topic>.typ` and compile with `typst compile reports/<file>.typ`.
//
// Usage:
//   #import "template.typ": *
//   #show: report.with(
//     title: "…",
//     date: "2026-10-06",
//     author: "…",
//     summary: [ … ],   // the "bottom line" box on page 1
//   )
//
// Helpers: status tags (`#verified[…]`, `#concluded`, `#assumed`, `#decided`) mirror the
// epistemic markers of the project memory (CLAUDE.md); `#finding`, `#caveat` and `#note` are
// callout boxes; `#runid` formats WandB run ids; `#tbl` is a compact table with a header row
// (short tables are kept on one page; pass `breakable: true` for a long-celled table).

#let palette = (
  accent: rgb("#2f5d8a"),
  muted: rgb("#5b6470"),
  rule: rgb("#c9cfd6"),
  header-fill: rgb("#eef2f6"),
  verified: rgb("#2e7d4f"),
  concluded: rgb("#2f5d8a"),
  assumed: rgb("#a86400"),
  decided: rgb("#6a3d9a"),
  finding-bg: rgb("#eef4fb"),
  caveat-bg: rgb("#fdf3e6"),
  note-bg: rgb("#f3f4f6"),
)

#let _tag(label, color, detail) = box(
  inset: (x: 3pt, y: 1pt),
  outset: (y: 1pt),
  radius: 2pt,
  stroke: 0.5pt + color,
  text(size: 0.78em, fill: color, weight: "medium")[#label#if detail != none [: #detail]],
)

/// Evidence status tags, as in the project memory. `detail` names the evidence,
/// e.g. `#verified[real ablation, one seed]`.
#let verified(detail) = _tag("verified", palette.verified, detail)
#let concluded = _tag("concluded", palette.concluded, none)
#let assumed = _tag("assumed", palette.assumed, none)
#let decided = _tag("decided", palette.decided, none)

#let _callout(title, body, bg, bar) = block(
  width: 100%,
  inset: (x: 10pt, y: 8pt),
  fill: bg,
  stroke: (left: 2.5pt + bar),
  breakable: true,
)[
  #if title != none [#text(weight: "bold", fill: bar)[#title] \ ]
  #body
]

#let finding(title: none, body) = _callout(title, body, palette.finding-bg, palette.accent)
#let caveat(title: "Caveat", body) = _callout(title, body, palette.caveat-bg, palette.assumed)
#let note(title: none, body) = _callout(title, body, palette.note-bg, palette.muted)

/// WandB run id in monospace, e.g. #runid("s-bd23f0d1").
#let runid(id) = raw(id)

/// Compact table: `columns` as in `table`, first row of `cells` is the header.
#let tbl(columns: auto, align: auto, breakable: auto, ..cells) = {
  let cells = cells.pos()
  let n = if type(columns) == int { columns } else { columns.len() }
  // Short tables stay on one page; long ones may break (the header row repeats).
  block(breakable: if breakable == auto { cells.len() > 12 * n } else { breakable }, table(
    columns: columns,
    align: if align == auto { (x, y) => if x == 0 { left } else { center } } else { align },
    stroke: (x, y) => (
      top: if y == 0 { 0.8pt + palette.muted } else if y == 1 { 0.5pt + palette.muted } else { 0.3pt + palette.rule },
      bottom: 0.8pt + palette.muted,
    ),
    fill: (x, y) => if y == 0 { palette.header-fill },
    inset: (x: 5pt, y: 4pt),
    table.header(..cells.slice(0, n).map(c => text(weight: "bold", c))),
    ..cells.slice(n),
  ))
}

#let report(
  title: none,
  subtitle: none,
  date: none,
  author: none,
  summary: none,
  body,
) = {
  set document(title: title, author: if author == none { () } else { author })
  set page(
    paper: "a4",
    margin: (x: 2.2cm, top: 2.2cm, bottom: 2.4cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 [
        #set text(size: 8pt, fill: palette.muted)
        #title #h(1fr) #date
      ]
    },
  )
  set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em, spacing: 0.9em)
  set heading(numbering: "1.1")
  show heading: set text(fill: palette.accent)
  show heading.where(level: 1): it => {
    v(0.6em)
    it
    v(0.2em)
  }
  show raw.where(block: false): set text(size: 0.92em)
  show link: set text(fill: palette.accent)
  set table(gutter: 0pt)
  show table: set text(size: 9pt)
  show table: set par(justify: false)
  show figure.caption: set text(size: 9pt)
  set list(indent: 0.6em)
  set enum(indent: 0.6em)

  // Title block
  block(width: 100%, below: 1.2em)[
    #set par(justify: false)
    #text(size: 18pt, weight: "bold", fill: palette.accent, hyphenate: false)[#title]
    #if subtitle != none [\ #text(size: 12pt, fill: palette.muted)[#subtitle]]
    #v(0.3em)
    #text(size: 9.5pt, fill: palette.muted)[
      #if date != none [Status as of #date]
      #if author != none [#h(0.6em)·#h(0.6em)#author]
    ]
  ]
  if summary != none {
    finding(title: "Bottom line", summary)
    v(0.6em)
  }
  body
}
