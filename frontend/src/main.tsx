import "./styles.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router";

import { Layout } from "./components/Layout";
import { ThemeContext } from "./components/ThemeContext";
import { Loading } from "./components/ui";
import { useTheme } from "./lib/theme";

const page = (load: () => Promise<Record<string, React.ComponentType>>, name: string) => {
  const Page = lazy(async () => ({ default: (await load())[name]! }));
  return (
    <Suspense fallback={<Loading rows={6} />}>
      <Page />
    </Suspense>
  );
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5 * 60_000, retry: 1, refetchOnWindowFocus: false },
  },
});

// Hash routing keeps deep links working on static hosts without rewrite rules.
const router = createHashRouter([
  {
    element: <Layout />,
    children: [
      { index: true, element: page(() => import("./pages/Overview"), "OverviewPage") },
      { path: "explorer", element: page(() => import("./pages/Explorer"), "ExplorerPage") },
      { path: "hierarchy", element: page(() => import("./pages/Hierarchy"), "HierarchyPage") },
      { path: "orders", element: page(() => import("./pages/Orders"), "OrdersPage") },
      { path: "models", element: page(() => import("./pages/Models"), "ModelsPage") },
      { path: "insights", element: page(() => import("./pages/Insights"), "InsightsPage") },
    ],
  },
]);

function App() {
  const theme = useTheme();
  return (
    <ThemeContext.Provider value={theme}>
      <RouterProvider router={router} />
    </ThemeContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
