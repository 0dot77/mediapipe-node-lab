# Kineforge

Node-based performance media prototyping environment for dancers and media artists.

## Brand Decisions

### 1) Logo Tone (Typography + Symbol)

- Wordmark tone: kinetic, industrial, precision-driven
- Recommended type direction: wide geometric sans + sharp cuts (e.g. "Sora", "Space Grotesk", "Eurostile-like")
- Symbol direction: `K` monogram fused with signal path / node edge motif
- Motion cue: subtle horizontal "motion streak" in `K` arm to imply movement tracking
- Color system: deep black graphite base + lime accent highlights

### 2) One-Line Product Description

`Kineforge is a node-based live media forge where body motion, AI vision, and GPU visuals are composed in real time.`

### 3) URL / Repository Naming Rules

- Canonical project slug: `kineforge`
- Product website: `kineforge.studio`
- Web app: `app.kineforge.studio`
- Docs: `docs.kineforge.studio`
- Core repository: `github.com/<org>/kineforge`
- Frontend/app repository (if split): `github.com/<org>/kineforge-web`
- Examples repository: `github.com/<org>/kineforge-examples`
- Packages scope (if published): `@kineforge/*`

## Current Stack

- Next.js (App Router)
- TypeScript
- Tailwind CSS v4
- React Flow (`@xyflow/react`)
- MediaPipe Tasks Vision (Face + Hand)

## Current Features

- Live webcam source node
- Face extraction node (landmarks + jaw metric)
- Hand extraction node (landmarks + pinch/lift metric)
- Landmark overlay node
- Gesture mapper node
- Stage output node with reactive visuals
- Per-node debug preview toggle
- Double-click node picker (material-editor-style)

## Run

```bash
git clone https://github.com/0dot77/kineforge.git
cd kineforge
npm install
npm run dev
```

Open `http://localhost:4173`

## Build

```bash
npm run build
npm run start
```

## Notes

- Camera permission is required.
- Models load from official MediaPipe model storage.
- If layout looks off, click `Reset Graph`.
