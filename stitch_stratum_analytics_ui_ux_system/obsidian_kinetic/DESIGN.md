---
name: Obsidian Kinetic
colors:
  surface: '#121315'
  surface-dim: '#121315'
  surface-bright: '#38393b'
  surface-container-lowest: '#0d0e10'
  surface-container-low: '#1b1c1e'
  surface-container: '#1f2022'
  surface-container-high: '#292a2c'
  surface-container-highest: '#343537'
  on-surface: '#e3e2e5'
  on-surface-variant: '#dbc2b0'
  inverse-surface: '#e3e2e5'
  inverse-on-surface: '#303033'
  outline: '#a38c7c'
  outline-variant: '#554336'
  surface-tint: '#ffb77d'
  primary: '#ffb77d'
  on-primary: '#4d2600'
  primary-container: '#d97707'
  on-primary-container: '#432100'
  inverse-primary: '#904d00'
  secondary: '#89ceff'
  on-secondary: '#00344d'
  secondary-container: '#00a2e6'
  on-secondary-container: '#00344e'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#00a572'
  on-tertiary-container: '#00311f'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdcc3'
  primary-fixed-dim: '#ffb77d'
  on-primary-fixed: '#2f1500'
  on-primary-fixed-variant: '#6e3900'
  secondary-fixed: '#c9e6ff'
  secondary-fixed-dim: '#89ceff'
  on-secondary-fixed: '#001e2f'
  on-secondary-fixed-variant: '#004c6e'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#121315'
  on-background: '#e3e2e5'
  surface-variant: '#343537'
typography:
  display-xl:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: 56px
    letterSpacing: -0.03em
  display-lg:
    fontFamily: Geist
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.025em
  display-lg-mobile:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.015em
  body-lg:
    fontFamily: Geist
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: -0.005em
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  metric-xl:
    fontFamily: JetBrains Mono
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.03em
  metric-md:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.02em
  metric-sm:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.06em
  caption:
    fontFamily: Geist
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit-2xs: 2px
  unit-xs: 4px
  unit-sm: 8px
  unit-md: 12px
  unit-base: 16px
  unit-lg: 20px
  unit-xl: 24px
  unit-2xl: 32px
  unit-3xl: 48px
  gutter-compact: 12px
  gutter-default: 16px
  gutter-spacious: 24px
---

## Brand & Style

This design system embodies the ethos of precision instrumentation for modern data intelligence. Rooted in analytical restraint, the interface avoids decorative excess, flashy neon glows, and performative rounded bubbles. Instead, it mirrors high-end physical hardware, cryptographic terminals, and specialized telemetry systems.

### Core Tenets
- **Analytical Restraint:** Visual weight is awarded strictly through information density, contrast, and tabular geometry rather than decorative flourishes.
- **Instrument-Grade Precision:** Every structural division relies on 1px hairline rules, disciplined micro-spacing, and exact monospaced metric alignment.
- **Intentional Contrast:** 90% of the canvas consists of deep slate, obsidian surfaces, and neutral typography, reserving chromatic color solely for state transitions, anomaly detections, and retention risk thresholds.
- **Editorial Legibility:** Long-form intelligence reports and analytical breakdowns blend editorial narrative rigor with structured data grids.

## Colors

The palette is engineered for prolonged operational focus in low-light and ambient control environments. The baseline environment is built on deep slate-obsidian substrates, avoiding pure pitch-black (`#000000`) to eliminate harsh eye fatigue and ensure fine contrast preservation across surface boundaries.

### Canvas & Surface Architecture
- **Base Canvas (`bg-canvas`):** `#090A0C` — The foundational viewport background.
- **Surface Low (`bg-surface-low`):** `#0F1115` — Recessed bays, inactive table rows, navigation tracks.
- **Surface Mid (`bg-surface-mid`):** `#15181E` — Primary analytical cards, metric modules, sidebars.
- **Surface High (`bg-surface-high`):** `#1C2027` — Modals, dropdown popovers, elevated context panels.
- **Surface Hover (`bg-surface-hover`):** `#242932` — Interactive element resting states upon pointer hover.

### Structural Hairlines & Borders
- **Hairline Subtle:** `rgba(255, 255, 255, 0.07)` (`#262C36` equivalent on Mid surfaces).
- **Hairline Strong:** `rgba(255, 255, 255, 0.14)` (`#313845` equivalent) for active component boundaries and focus outlines.
- **Hairline Accent:** `rgba(217, 119, 6, 0.35)` for selective risk focal triggers.

### Text & Glyph Hierarchy
- **Text High-Contrast:** `#F1F3F7` — Metric outputs, prominent headlines, primary values.
- **Text Body / Secondary:** `#A9B4C4` — Analytical descriptions, table cells, contextual reports.
- **Text Muted:** `#8590A2` — Column headers, metadata keys, timestamps, inactive states.
- **Text Ghost:** `#4A5464` — De-emphasized units, placeholders, micro-dividers.

### Semantic Telemetry & Intelligence Tiers
- **Amber / Bronze Accent (`#D97706` / `#F59E0B`):** Actionable retention interventions, high priority queue items, moderate churn warnings.
- **Controlled Brick (`#EF4444`):** Severe churn probability (>75%), revenue leakage, metric regression.
- **Calm Emerald (`#10B981`):** Expansion velocity, low churn probability, retained value health.
- **Slate Teal / Cyan (`#0EA5E9`):** Algorithmic confidence scores, system heuristics, clustering models.
- **Deep Slate Violet (`#8B5CF6`):** Enterprise account cohorts and Lifetime Value (LTV) anomalies.

## Typography

Typography functions as the structural scaffolding of the analytical workspace. The pairing combines the geometric restraint of Geist with the computational rigor of monospaced figures.

### Typographic Guidelines
- **Tabular Numerics:** All currency figures, percentages, churn rates, and timestamps must leverage tabular numerals (`tnum`) and slashed zeros where supported, enforcing vertical scan lines down dense columns.
- **Metadata Framing:** Uppercase strings are exclusively set in `label-mono` with wide letter spacing (`+0.06em`), used to designate classification tags, data dimensions, table headers, and status badges.
- **Headline Restraint:** Metric values and card titles maintain compact vertical line heights to optimize information-per-pixel density without feeling cramped.

## Layout & Spacing

The layout is constructed on an adaptable 4px/8px mathematical cadence, optimizing screen real estate for multi-pane analytical flows.

### Layout Philosophy
- **Split-Pane Density:** Dashboards employ a 16-column flexible layout desktop grid or dynamic multi-rail workspace. Margins are fixed at 24px on desktop displays to maximize table and cohort canvas widths.
- **Asymmetrical Anchor:** Primary operational metrics anchor the left or upper deck, while detailed cohort drill-downs, risk timelines, and AI prompt interactions occupy flexible central-right panels.
- **Responsive Adaptations:**
  - **Desktop (>1440px):** 16 columns; 16px gutter; persistent secondary sidebar for customer contextual cards.
  - **Laptop (1024px - 1439px):** 12 columns; 16px gutter; collapsable sidebars; horizontal scroll preserved on data tables with fixed identifier columns.
  - **Tablet (768px - 1023px):** 8 columns; 12px gutter; split panes stack into tabbed card decks.
  - **Mobile (<767px):** 4 columns; 12px margins; navigation transitions to an off-canvas drawer or compact bottom dock; high-priority risk indicators compress to single-value summary rows.

## Elevation & Depth

Spatial layering in this design system is driven by tonal variation and crisp border transitions rather than diffused blurs or heavy drop shadows.

### Depth Strategy
- **Layer 0 (Canvas):** Tone `#090A0C`. Baseline viewport container.
- **Layer 1 (Recessed/Track):** Tone `#0F1115` bounded by 1px `rgba(255, 255, 255, 0.05)`. Used for sunken metric bands, graph axes, and filter bars.
- **Layer 2 (Primary Modules):** Tone `#15181E` bounded by 1px `rgba(255, 255, 255, 0.08)`. Default elevation for cards and data tables.
- **Layer 3 (Overlays & Contextuals):** Tone `#1C2027` with a direct hairline border `rgba(255, 255, 255, 0.12)`. Supported by a razor-thin shadow: `0 4px 20px -2px rgba(0, 0, 0, 0.65)`. Used for slide-over drawer inspectors, tooltips, and command palettes.
- **Zero-Shadow Bias:** Avoid ambient tinted colored glows beneath buttons or status tags. Contrast is earned via boundary hairline luminance.

## Shapes

The geometric form language is strict, grounded, and industrial:
- **Base Elements:** Buttons, chips, single-line form inputs, and status badges take a `6px` radius (`roundedness: 1`).
- **Containers:** Analytical modules, insight cards, and tabular panels take an `8px` radius (`rounded-lg: 0.5rem`).
- **Overlays:** Dialogs, command palettes, and drawer panels take a `10px` radius.
- **Prohibited Shapes:** Fully rounded pills (`9999px`) are strictly forbidden, as they introduce unnecessary softness to an analytical cockpit.

## Components

### 1. Buttons
- **Primary Action (Retention Interventions, Export Trigger):** Background `#D97706`; text `#090A0C` (weight 500); 1px solid `#F59E0B`. Hover: `#F59E0B`. Active: `#B45309`.
- **Secondary (Default Analytical Actions):** Background `#1C2027`; text `#F1F3F7`; 1px solid `rgba(255, 255, 255, 0.12)`. Hover: background `#242932`, border `rgba(255, 255, 255, 0.2)`.
- **Ghost / Utility:** Background transparent; text `#A9B4C4`; 1px solid transparent. Hover: background `rgba(255, 255, 255, 0.05)`, text `#F1F3F7`.
- **Padding & Geometry:** Small: `h-7`, `px-2.5`, `label-mono`; Medium: `h-9`, `px-3.5`, `body-sm` (font-weight 500).

### 2. Status Chips & Risk Badges
- Constructed with a structural background, 1px border, and leading monospaced status indicator dot.
- **Acute Churn Risk:** Background `rgba(239, 68, 68, 0.10)`; border `rgba(239, 68, 68, 0.3)`; text `#F87171`.
- **Action Required / Moderate Risk:** Background `rgba(217, 119, 6, 0.12)`; border `rgba(245, 158, 11, 0.3)`; text `#FBBF24`.
- **Stable / Expanded Cohort:** Background `rgba(16, 185, 129, 0.10)`; border `rgba(16, 185, 129, 0.25)`; text `#34D399`.
- **Structure:** `label-mono`, height `22px`, horizontal padding `8px`, radius `4px`.

### 3. Metric & Telemetry Cards
- **Container:** Background `#15181E`; border 1px solid `rgba(255, 255, 255, 0.07)`; padding `16px`.
- **Layout:** Metric key in uppercase `label-mono` (`#8590A2`) top-left; delta badge top-right. Main numeric reading in `metric-xl` (`#F1F3F7`) with tabular positioning. Footer contains trailing 30-day sparkline or sample baseline comparison in `caption` (`#8590A2`).

### 4. Data Tables (Cohort & Churn Prioritization)
- **Header Row:** Height `32px`; background `#0F1115`; text `label-mono` (`#8590A2`); border-bottom 1px solid `rgba(255, 255, 255, 0.10)`.
- **Data Rows:** Height `44px`; alternating cell borders 1px solid `rgba(255, 255, 255, 0.04)`; font `body-md` (`#A9B4C4`). Numerical columns right-aligned with `metric-sm` typography.
- **Hover State:** Entire row highlights to `#1C2027` with a 2px vertical amber indicator mark anchored to the leading cell.

### 5. Input Fields & Query Terminals
- **Structure:** Background `#0F1115`; border 1px solid `rgba(255, 255, 255, 0.10)`; text `#F1F3F7`; typography `body-md`; radius `6px`.
- **Focus:** Border `#D97706`; subtle ring `0 0 0 1px rgba(217, 119, 6, 0.35)`.
- **Natural Language / AI Prompt Bar:** Height `48px`; background `#15181E`; leading indicator glyph in `#D97706`; trailing shortcut indicator (e.g., `⌘K`) in `label-mono` (`#8590A2`).

### 6. Specialized Components: Risk Heatstrip & Confidence Indicators
- **Risk Heatstrip:** A segmented, 4px-tall horizontal telemetry bar representing churn vector scores from 0 to 100, broken into discrete micro-cells with 1px gaps, shifting from `#10B981` through `#D97706` to `#EF4444`.
- **Model Confidence Meter:** Monospaced fractional rating (e.g., `P(CHURN) = 0.84`) accompanied by an inline hairline confidence interval gauge.