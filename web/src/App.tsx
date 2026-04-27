/**
 * Purpose: App shell for the AuralMind2 premium web workspace.
 * Data shapes: delegates mastering session state to MasteringWorkspace, which
 * consumes API DTOs from src/lib/api.ts and local preview state.
 * Syntax: imported by src/main.tsx and rendered as <App />.
 * Important functions: App at line 12.
 * Possible bugs: future routes may need React Router once deep links are added.
 * Enhance next: add a /sessions/:id route; expose a ChatGPT Apps widget shell.
 */

import { MasteringWorkspace } from './features/mastering/MasteringWorkspace'
import './styles/theme.css'
import './styles/app.css'

function App() {
  return <MasteringWorkspace />
}

export default App
