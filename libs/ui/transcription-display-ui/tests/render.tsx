import type { ReactNode } from 'react';

import { ThemeProvider, createTheme } from '@mui/material/styles';

import { render } from '@testing-library/react';

import { TranscriptionDisplayHeightContext } from '#src/contexts/transcription-display-height-context.js';

// A theme that provides the custom `transcriptionColor` palette entry the
// caption components read (IconButton/Typography `color="transcriptionColor"`
// would otherwise throw looking up `palette.transcriptionColor.main`).
const testTheme = createTheme({
  palette: { transcriptionColor: { main: '#ffff00' } },
});

interface Options {
  containerHeightPx?: number;
}

/**
 * Renders caption UI inside the MUI theme + the display-height context both the
 * container and the bounded preference controls depend on.
 */
export function withProviders(
  ui: ReactNode,
  { containerHeightPx = 600 }: Options = {},
) {
  return (
    <ThemeProvider theme={testTheme}>
      <TranscriptionDisplayHeightContext.Provider
        value={{ containerHeightPx, setContainerHeightPx: () => undefined }}
      >
        {ui}
      </TranscriptionDisplayHeightContext.Provider>
    </ThemeProvider>
  );
}

export function renderWithProviders(ui: ReactNode, options: Options = {}) {
  return render(withProviders(ui, options));
}
