import { createContext } from "react";

import type { ThemeChoice } from "../lib/theme";

export interface ThemeState {
  choice: ThemeChoice;
  /** The scheme in effect; charts re-read CSS tokens when it changes. */
  resolved: "light" | "dark";
  cycle: () => void;
}

export const ThemeContext = createContext<ThemeState>({
  choice: "system",
  resolved: "light",
  cycle: () => undefined,
});
