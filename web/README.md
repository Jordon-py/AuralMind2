# AuralMind2 Web Workspace

Premium React interface for the AuralMind2 mastering server. The UI is a calm, production-oriented workspace for selecting audio, choosing a mastering profile, monitoring metrics, and preparing delivery formats.

## Project Understanding

AuralMind2 already has a Python FastMCP server and a local Flask dashboard. This `web/` package adds a separate modern frontend without changing the server runtime. In local development, Vite proxies `/api/*` requests to the existing Flask dashboard at `http://127.0.0.1:5000`.

## UI Concept Summary

The interface uses a three-zone command center:

- left setup rail for source audio, profile, stem mode, and start action
- center live analysis area with waveform, meters, and spectrum
- right queue and delivery panel for status and export formats

The visual system is intentionally restrained: warm canvas, charcoal type, teal action states, 8px radii, quiet borders, and accessible focus states.

## Tech Stack

- React + TypeScript for componentized UI
- Vite for fast local development and static builds
- Custom CSS tokens for a small, explicit design system
- Lucide icons for consistent interface symbols
- Playwright for browser smoke checks

This is deliberately not overbuilt with a global state manager. The app keeps one local session state and a typed API client.

## Folder Structure

```text
web/
  index.html
  package.json
  playwright.config.ts
  vite.config.ts
  src/
    App.tsx
    main.tsx
    index.css
    features/mastering/
      MasteringWorkspace.tsx
      masteringData.ts
      masteringTypes.ts
    lib/
      api.ts
    styles/
      theme.css
      app.css
  tests/
    mastering-workspace.spec.ts
```

## Local Setup

Install frontend dependencies:

```bash
cd web
npm install
```

Run only the frontend:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Run with the existing Flask dashboard API:

```bash
# terminal 1, repo root
python run_ui.py --server

# terminal 2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# X: 2D array or DataFrame, y: 1D labels
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = HistGradientBoostingClassifier(
    learning_rate=0.05,      # smaller LR + more trees is a common pattern
    max_iter=300,            # number of boosting iterations (trees)
    max_leaf_nodes=31,       # tree size / complexity
    l2_regularization=1.0,   # regularization to fight overfitting
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

clf.fit(X_train, y_train)

# Class predictions
y_pred = clf.predict(X_valid)

# Probabilities (for AUC, Brier, calibration etc.)
proba_pos = clf.predict_proba(X_valid)[:, 1]

```

The Vite dev server proxies `/api` to `http://127.0.0.1:5000`.

## Verification

```bash
cd web
npm run lint
npm run build
npx playwright install chromium
npm run test:e2e
```

The Playwright smoke test verifies the desktop workflow, preview fallback, and mobile horizontal overflow.

## Deployment

Build the static frontend:

```bash
cd web
npm run build
```

The static output is written to:

```text
web/dist/
```

If the frontend and API are on different origins, set:

```text
VITE_AURALMIND_API_BASE=https://your-api-host.example.com
```

Then rebuild.

## Horizon Assessment

`https://horizon.prefect.io` is not a generic static frontend host. Prefect Horizon is positioned for managed MCP server deployment. It is a better match for the AuralMind2 MCP server surface than for this React static frontend.

Recommended split:

- deploy the MCP server through Horizon if you want a managed MCP target
- deploy this `web/` static frontend through a frontend host such as Vercel, Netlify, Cloudflare Pages, or Render Static Site
- deploy the Python API/Flask or ASGI service separately if the frontend needs live mastering calls

## GitHub Readiness

Commit `web/` with:

- `package.json`
- `package-lock.json`
- `src/`
- `tests/`
- config files

Do not commit:

- `node_modules/`
- `dist/`
- Playwright screenshots or traces

Both ignored paths are already covered by `web/.gitignore`.
