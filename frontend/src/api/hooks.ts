import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { getJson } from "./client";
import type {
  Analysis,
  HierarchyDetail,
  ModelRun,
  OrderLine,
  Overview,
  Page,
  PolicyResult,
  QualityCheck,
  Score,
  SeriesDetail,
  SeriesSummary,
  Store,
} from "./types";

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: ({ signal }) => getJson<Overview>("/overview", {}, signal) });

export const useStores = () =>
  useQuery({ queryKey: ["stores"], queryFn: ({ signal }) => getJson<Store[]>("/stores", {}, signal) });

export const useFamilies = () =>
  useQuery({ queryKey: ["families"], queryFn: ({ signal }) => getJson<string[]>("/families", {}, signal) });

export interface SeriesFilter {
  store_nbr?: number;
  family?: string;
  item_nbr?: number;
  page: number;
  page_size: number;
}

export const useSeries = (filter: SeriesFilter) =>
  useQuery({
    queryKey: ["series", filter],
    queryFn: ({ signal }) => getJson<Page<SeriesSummary>>("/series", { ...filter }, signal),
    placeholderData: keepPreviousData,
  });

export const useSeriesDetail = (store: number | undefined, item: number | undefined) =>
  useQuery({
    queryKey: ["series-detail", store, item],
    queryFn: ({ signal }) => getJson<SeriesDetail>(`/series/${store}/${item}`, {}, signal),
    enabled: store !== undefined && item !== undefined,
  });

export const useHierarchy = (store: number, family: string) =>
  useQuery({
    queryKey: ["hierarchy", store, family],
    queryFn: ({ signal }) =>
      getJson<HierarchyDetail>(`/hierarchy/${store}/${encodeURIComponent(family)}`, {}, signal),
  });

export interface OrderFilter {
  store_nbr?: number;
  family?: string;
  page: number;
  page_size: number;
}

export const useOrders = (filter: OrderFilter) =>
  useQuery({
    queryKey: ["orders", filter],
    queryFn: ({ signal }) => getJson<Page<OrderLine>>("/orders", { ...filter }, signal),
    placeholderData: keepPreviousData,
  });

export const usePolicies = (segment?: string) =>
  useQuery({
    queryKey: ["policies", segment],
    queryFn: ({ signal }) => getJson<PolicyResult[]>("/policies", { segment }, signal),
  });

export const useModels = (level?: "item" | "family") =>
  useQuery({
    queryKey: ["models", level],
    queryFn: ({ signal }) => getJson<ModelRun[]>("/models", { level }, signal),
  });

export const useRunScores = (runId: string | undefined) =>
  useQuery({
    queryKey: ["scores", runId],
    queryFn: ({ signal }) => getJson<Score[]>(`/models/${runId}/scores`, {}, signal),
    enabled: Boolean(runId),
  });

export const useAnalyses = () =>
  useQuery({ queryKey: ["analyses"], queryFn: ({ signal }) => getJson<Analysis[]>("/analyses", {}, signal) });

export const useQuality = () =>
  useQuery({ queryKey: ["quality"], queryFn: ({ signal }) => getJson<QualityCheck[]>("/quality", {}, signal) });
