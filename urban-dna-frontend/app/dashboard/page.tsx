"use client";

import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import Papa from "papaparse";
import { DeckGL } from "@deck.gl/react";
import { ColumnLayer, ScatterplotLayer } from "@deck.gl/layers";
import Map from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import Link from "next/link";
import {
    Dna, ArrowLeft, Download, Layers, X, Info,
    Search, Globe, AlertTriangle, RefreshCw,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type BuildingPoint = {
    lat: number;
    lon: number;
    cluster_id: number;
    category?: string;
    area_m2?: number;
    perimeter_m?: number;
    shape_index?: number;
    risk_score?: number;
    [key: string]: unknown;
};

type ApiResponse = {
    city: string;
    total: number;
    source: "precomputed" | "synthetic";
    elapsed_ms: number;
    clusters: Record<string, number>;
    points: BuildingPoint[];
};

// ── Constants ─────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const INITIAL_VIEW_STATE = {
    longitude: 30.06,
    latitude: -1.94,
    zoom: 13,
    pitch: 50,
    bearing: -10,
};

const CLUSTER_META: Record<number, { label: string; color: [number, number, number]; hex: string; desc: string }> = {
    0: { label: "Informal / High Risk", color: [248, 113, 113], hex: "#f87171", desc: "Small, dense, irregular structures — likely informal settlements needing urgent attention." },
    1: { label: "Upgrading Zone", color: [251, 191, 36], hex: "#fbbf24", desc: "Transitional morphology — existing structure but lacking formal grid. Upgrade pathway viable." },
    2: { label: "Stable / Formal", color: [45, 212, 191], hex: "#2dd4bf", desc: "Larger footprints, regular spacing, formal building patterns. Low intervention priority." },
};

// Map a category string back to a cluster_id (for API responses that send category name)
function categoryToId(cat?: string): number {
    if (!cat) return 2;
    if (cat.includes("Informal") || cat.includes("High")) return 0;
    if (cat.includes("Upgrading")) return 1;
    return 2;
}

function getClusterMeta(id: number) {
    return CLUSTER_META[id] ?? CLUSTER_META[2];
}

// ── Geocode helper (client-side, uses Nominatim) ──────────────────────────────

async function geocodeCity(query: string): Promise<{ lat: number; lon: number } | null> {
    try {
        const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`;
        const res = await fetch(url, { headers: { "Accept-Language": "en" } });
        const data = await res.json();
        if (data.length > 0) {
            return { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon) };
        }
    } catch { /* silent */ }
    return null;
}

// Compute bounding box around a centre point (in degrees, ~city scale)
function bboxAround(lat: number, lon: number, deg = 0.05) {
    return {
        min_lat: lat - deg, max_lat: lat + deg,
        min_lon: lon - deg, max_lon: lon + deg,
    };
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function Dashboard() {
    const [points, setPoints] = useState<BuildingPoint[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedPoint, setSelectedPoint] = useState<BuildingPoint | null>(null);
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
    const [activeLayer, setActiveLayer] = useState<"scatter" | "column">("column");
    const [filterCluster, setFilterCluster] = useState<number | null>(null);
    const [cityName, setCityName] = useState("Kigali, Rwanda");
    const [searchInput, setSearchInput] = useState("");
    const [apiSource, setApiSource] = useState<"precomputed" | "synthetic" | "csv">("csv");
    const [elapsedMs, setElapsedMs] = useState<number | null>(null);

    const abortRef = useRef<AbortController | null>(null);

    // ── Fetch from API ─────────────────────────────────────────────────────────
    const fetchFromApi = useCallback(async (lat: number, lon: number, city: string) => {
        // Cancel any in-flight request
        if (abortRef.current) abortRef.current.abort();
        const ctrl = new AbortController();
        abortRef.current = ctrl;

        setLoading(true);
        setError(null);
        setSelectedPoint(null);

        const bbox = bboxAround(lat, lon, 0.06);
        const body = {
            ...bbox,
            city: city.toLowerCase().split(",")[0].trim(),
            max_points: 2000,
        };

        try {
            const res = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: ctrl.signal,
            });

            if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
            const data: ApiResponse = await res.json();

            const cleaned: BuildingPoint[] = data.points
                .filter(p => typeof p.lat === "number" && typeof p.lon === "number")
                .map(p => ({
                    ...p,
                    cluster_id: typeof p.cluster_id === "number"
                        ? p.cluster_id
                        : categoryToId(p.category),
                }));

            setPoints(cleaned);
            setApiSource(data.source as "precomputed" | "synthetic");
            setElapsedMs(data.elapsed_ms);
        } catch (e: unknown) {
            if ((e as Error).name === "AbortError") return;
            // Fallback: try static CSV
            await fetchCsvFallback(ctrl.signal);
        } finally {
            setLoading(false);
        }
    }, []);

    // ── CSV fallback (original behaviour) ─────────────────────────────────────
    const fetchCsvFallback = useCallback(async (signal?: AbortSignal) => {
        try {
            const res = await fetch("/kigali_results.csv", { signal });
            if (!res.ok) throw new Error("CSV not found");
            const text = await res.text();
            const parsed = Papa.parse<Record<string, unknown>>(text, {
                header: true, dynamicTyping: true, skipEmptyLines: true,
            });
            const cleaned: BuildingPoint[] = (parsed.data || [])
                .filter(r => typeof r.lat === "number" && typeof r.lon === "number")
                .map(r => ({
                    ...r,
                    lat: r.lat as number,
                    lon: r.lon as number,
                    cluster_id: isNaN(Number(r.cluster_id)) ? 0 : Number(r.cluster_id),
                }));
            setPoints(cleaned);
            setApiSource("csv");
            setError(null);
        } catch (e: unknown) {
            if ((e as Error).name !== "AbortError") {
                setError("Could not load data. Make sure the API (port 8000) or kigali_results.csv is available.");
            }
        } finally {
            setLoading(false);
        }
    }, []);

    // ── Initial load ───────────────────────────────────────────────────────────
    useEffect(() => {
        fetchFromApi(INITIAL_VIEW_STATE.latitude, INITIAL_VIEW_STATE.longitude, "kigali");
    }, [fetchFromApi]);

    // ── City search ────────────────────────────────────────────────────────────
    const handleSearch = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchInput.trim()) return;
        setLoading(true);

        const coords = await geocodeCity(searchInput);
        if (!coords) {
            setError(`Could not find "${searchInput}". Try a more specific location.`);
            setLoading(false);
            return;
        }

        setCityName(searchInput);
        setSearchInput("");
        setViewState(vs => ({
            ...vs,
            latitude: coords.lat,
            longitude: coords.lon,
            zoom: 13,
        }));
        await fetchFromApi(coords.lat, coords.lon, searchInput);
    }, [searchInput, fetchFromApi]);

    // ── Stats ──────────────────────────────────────────────────────────────────
    const stats = useMemo(() => {
        const total = points.length;
        const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0 };
        points.forEach(p => { counts[p.cluster_id] = (counts[p.cluster_id] ?? 0) + 1; });
        return { total, counts };
    }, [points]);

    // ── Filtered data ──────────────────────────────────────────────────────────
    const displayPoints = useMemo(
        () => filterCluster !== null ? points.filter(p => p.cluster_id === filterCluster) : points,
        [points, filterCluster]
    );

    // ── Layers ─────────────────────────────────────────────────────────────────
    const layers = useMemo(() => {
        if (activeLayer === "column") {
            return [new ColumnLayer<BuildingPoint>({
                id: "column",
                data: displayPoints,
                diskResolution: 6,
                radius: 14,
                elevationScale: 4,
                getPosition: d => [d.lon, d.lat],
                getElevation: d => (d.cluster_id === 0 ? 8 : d.cluster_id === 1 ? 5 : 3),
                getFillColor: d => [...getClusterMeta(d.cluster_id).color, 200] as [number, number, number, number],
                pickable: true,
                autoHighlight: true,
            })];
        }
        return [new ScatterplotLayer<BuildingPoint>({
            id: "scatter",
            data: displayPoints,
            radiusMinPixels: 3,
            radiusMaxPixels: 9,
            getPosition: d => [d.lon, d.lat],
            getFillColor: d => [...getClusterMeta(d.cluster_id).color, 220] as [number, number, number, number],
            pickable: true,
            autoHighlight: true,
        })];
    }, [displayPoints, activeLayer]);

    const handleClick = useCallback((info: { object?: BuildingPoint | null; coordinate?: [number, number] } | null) => {
        if (info?.object) {
            setSelectedPoint(info.object);
        } else {
            setSelectedPoint(null);
        }
    }, []);

    // ── Export ─────────────────────────────────────────────────────────────────
    const handleExport = () => {
        const csv = Papa.unparse(points);
        const blob = new Blob([csv], { type: "text/csv" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `urban_dna_${cityName.replace(/[^a-z0-9]/gi, "_").toLowerCase()}.csv`;
        a.click();
    };

    const meta = selectedPoint ? getClusterMeta(selectedPoint.cluster_id) : null;

    // ── Source badge colour ────────────────────────────────────────────────────
    const sourceBadge = { precomputed: { color: "#2dd4bf", label: "● LIVE DATA" }, synthetic: { color: "#fbbf24", label: "◐ AI SYNTHETIC" }, csv: { color: "#818cf8", label: "▣ LOCAL CSV" } }[apiSource];

    return (
        <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden", background: "#020617" }}>

            {/* ── MAP ────────────────────────────────────────────────────────── */}
            <DeckGL
                layers={layers}
                initialViewState={INITIAL_VIEW_STATE}
                controller
                viewState={viewState}
                onViewStateChange={({ viewState: vs }: { viewState: typeof INITIAL_VIEW_STATE }) => setViewState(vs)}
                onClick={handleClick}
            >
                <Map
                    reuseMaps
                    mapLib={maplibregl}
                    mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    style={{ width: "100%", height: "100%" }}
                />
            </DeckGL>

            {/* ── TOP BAR ──────────────────────────────────────────────────────── */}
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0, zIndex: 30,
                padding: "0 16px", height: 56,
                display: "flex", alignItems: "center", justifyContent: "space-between",
                background: "rgba(2,6,23,0.88)", backdropFilter: "blur(16px)",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                gap: 12,
            }}>
                {/* Left: back + title */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                    <Link href="/" style={{ display: "flex", alignItems: "center", gap: 5, color: "#64748b", textDecoration: "none", fontSize: "0.78rem" }}>
                        <ArrowLeft size={13} /> Back
                    </Link>
                    <span style={{ color: "rgba(255,255,255,0.12)" }}>|</span>
                    <Dna size={15} color="#2dd4bf" />
                    <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.75rem", fontWeight: 700, color: "#f8fafc", whiteSpace: "nowrap" }}>
                        Dashboard · {cityName}
                    </span>
                    {/* Source badge */}
                    <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.58rem", color: sourceBadge.color, letterSpacing: "0.1em" }}>
                        {sourceBadge.label}
                    </span>
                    {elapsedMs && (
                        <span style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.55rem", color: "#334155" }}>
                            {elapsedMs.toFixed(0)}ms
                        </span>
                    )}
                </div>

                {/* Centre: city search */}
                <form onSubmit={handleSearch} style={{ display: "flex", gap: 6, flex: 1, maxWidth: 400 }}>
                    <div style={{ display: "flex", flex: 1, alignItems: "center", gap: 8, background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)", borderRadius: 10, padding: "0 12px" }}>
                        <Globe size={13} color="#64748b" />
                        <input
                            type="text"
                            value={searchInput}
                            onChange={e => setSearchInput(e.target.value)}
                            placeholder="Search any city… (e.g. Nairobi, Lagos)"
                            style={{
                                all: "unset", flex: 1, fontSize: "0.75rem", color: "#f8fafc",
                                fontFamily: "Inter, sans-serif",
                                "::placeholder": { color: "#334155" } as React.CSSProperties,
                            }}
                        />
                    </div>
                    <button type="submit" disabled={loading} style={{
                        all: "unset", cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
                        padding: "6px 12px", borderRadius: 8, fontSize: "0.72rem",
                        background: "rgba(45,212,191,0.15)", border: "1px solid rgba(45,212,191,0.3)",
                        color: "#2dd4bf", fontWeight: 600,
                        opacity: loading ? 0.5 : 1,
                    }}>
                        {loading ? <RefreshCw size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Search size={12} />}
                        Analyse
                    </button>
                </form>

                {/* Right: layer toggle + export */}
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                    <div style={{ display: "flex", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, overflow: "hidden" }}>
                        {(["column", "scatter"] as const).map(l => (
                            <button key={l} onClick={() => setActiveLayer(l)} style={{
                                all: "unset", cursor: "pointer", padding: "6px 12px", fontSize: "0.68rem",
                                fontFamily: "'Space Mono', monospace", letterSpacing: "0.06em", textTransform: "uppercase",
                                background: activeLayer === l ? "rgba(45,212,191,0.15)" : "transparent",
                                color: activeLayer === l ? "#2dd4bf" : "#64748b",
                                transition: "all 0.2s",
                            }}>
                                {l === "column" ? "3D" : "Dot"}
                            </button>
                        ))}
                    </div>
                    <button onClick={handleExport} disabled={points.length === 0} style={{
                        all: "unset", cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
                        padding: "6px 12px", borderRadius: 8, fontSize: "0.68rem",
                        background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)",
                        color: "#94a3b8",
                    }}>
                        <Download size={12} /> Export CSV
                    </button>
                </div>
            </div>

            {/* ── KPI CARDS (top-right) ─────────────────────────────────────────── */}
            <div style={{ position: "absolute", top: 68, right: 16, zIndex: 20, display: "flex", flexDirection: "column", gap: 8, width: 196 }}>
                <div style={{ background: "rgba(2,6,23,0.88)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 14, padding: "12px 16px" }}>
                    <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.56rem", color: "#64748b", letterSpacing: "0.2em", marginBottom: 4 }}>TOTAL BUILDINGS</div>
                    <div style={{ fontSize: "1.6rem", fontWeight: 800, fontFamily: "'Space Mono', monospace", color: "#f8fafc" }}>
                        {loading ? "…" : stats.total.toLocaleString()}
                    </div>
                </div>
                {Object.entries(CLUSTER_META).map(([id, m]) => (
                    <div
                        key={id}
                        onClick={() => setFilterCluster(filterCluster === Number(id) ? null : Number(id))}
                        style={{
                            background: filterCluster === Number(id) ? m.hex + "18" : "rgba(2,6,23,0.88)",
                            backdropFilter: "blur(12px)",
                            border: `1px solid ${filterCluster === Number(id) ? m.hex + "55" : "rgba(255,255,255,0.08)"}`,
                            borderRadius: 14, padding: "10px 16px", cursor: "pointer",
                            transition: "all 0.2s",
                            boxShadow: filterCluster === Number(id) ? `0 0 20px ${m.hex}22` : "none",
                        }}
                    >
                        <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.52rem", color: m.hex, letterSpacing: "0.14em", marginBottom: 3 }}>
                            {m.label.toUpperCase()}
                        </div>
                        <div style={{ fontSize: "1.15rem", fontWeight: 800, fontFamily: "'Space Mono', monospace", color: m.hex }}>
                            {loading ? "…" : (stats.counts[Number(id)] ?? 0).toLocaleString()}
                        </div>
                        <div style={{ fontSize: "0.62rem", color: "#64748b", marginTop: 2 }}>
                            {stats.total > 0 ? (((stats.counts[Number(id)] ?? 0) / stats.total) * 100).toFixed(1) + "%" : "-"}
                        </div>
                    </div>
                ))}
                {filterCluster !== null && (
                    <button onClick={() => setFilterCluster(null)} style={{
                        all: "unset", cursor: "pointer", textAlign: "center",
                        padding: "7px", borderRadius: 10, fontSize: "0.66rem",
                        background: "rgba(255,255,255,0.05)", color: "#64748b",
                        border: "1px solid rgba(255,255,255,0.08)",
                    }}>
                        Show all clusters
                    </button>
                )}
            </div>

            {/* ── BUILDING DNA SIDEBAR ──────────────────────────────────────────── */}
            <aside style={{
                position: "absolute", left: 16, top: 68, zIndex: 20, width: 292,
                background: "rgba(2,6,23,0.92)", backdropFilter: "blur(16px)",
                border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20,
                padding: "22px", color: "#f8fafc",
                transform: selectedPoint ? "translateX(0)" : "translateX(-340px)",
                transition: "transform 0.35s cubic-bezier(0.4,0,0.2,1)",
                boxShadow: meta ? `0 0 40px ${meta.hex}18` : "none",
            }}>
                {selectedPoint && meta ? (
                    <>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                            <div>
                                <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.56rem", color: "#64748b", letterSpacing: "0.2em", marginBottom: 5 }}>
                                    BUILDING DNA PROFILE
                                </div>
                                <span style={{
                                    display: "inline-block", padding: "3px 11px", borderRadius: 999,
                                    background: meta.hex + "20", border: `1px solid ${meta.hex}40`,
                                    color: meta.hex, fontSize: "0.68rem", fontWeight: 700,
                                    fontFamily: "'Space Mono', monospace",
                                }}>
                                    {meta.label}
                                </span>
                            </div>
                            <button onClick={() => setSelectedPoint(null)} style={{ all: "unset", cursor: "pointer", color: "#64748b", padding: 4 }}>
                                <X size={15} />
                            </button>
                        </div>

                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            <div style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 11, padding: "10px 12px" }}>
                                <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.54rem", color: "#64748b", letterSpacing: "0.15em", marginBottom: 5 }}>COORDINATES</div>
                                <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.76rem", color: "#94a3b8" }}>
                                    {selectedPoint.lat.toFixed(5)}, {selectedPoint.lon.toFixed(5)}
                                </div>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                                {(["area_m2", "perimeter_m", "shape_index", "risk_score"] as const).map(k =>
                                    selectedPoint[k] != null && (
                                        <div key={k} style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 11, padding: "9px 11px" }}>
                                            <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.5rem", color: "#64748b", letterSpacing: "0.12em", marginBottom: 3 }}>
                                                {k.replace(/_/g, " ").toUpperCase()}
                                            </div>
                                            <div style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.8rem", color: "#f8fafc" }}>
                                                {typeof selectedPoint[k] === "number" ? Number(selectedPoint[k]).toFixed(2) : String(selectedPoint[k])}
                                            </div>
                                        </div>
                                    )
                                )}
                            </div>

                            <div style={{ background: meta.hex + "10", border: `1px solid ${meta.hex}25`, borderRadius: 11, padding: "11px 13px", display: "flex", gap: 9 }}>
                                <Info size={13} color={meta.hex} style={{ flexShrink: 0, marginTop: 2 }} />
                                <p style={{ fontSize: "0.75rem", color: "#94a3b8", lineHeight: 1.6, margin: 0 }}>{meta.desc}</p>
                            </div>
                        </div>
                    </>
                ) : (
                    <div style={{ color: "#64748b", fontSize: "0.78rem", textAlign: "center" }}>
                        <Layers size={22} style={{ margin: "0 auto 10px", opacity: 0.4 }} />
                        Click any building to read its DNA profile
                    </div>
                )}
            </aside>

            {/* ── LOADING ───────────────────────────────────────────────────────── */}
            {loading && (
                <div style={{ position: "absolute", inset: 0, zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(2,6,23,0.72)", backdropFilter: "blur(8px)" }}>
                    <div style={{ textAlign: "center" }}>
                        <Dna size={38} color="#2dd4bf" style={{ margin: "0 auto 14px" }} />
                        <p style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.78rem", color: "#64748b" }}>
                            Sequencing urban DNA…
                        </p>
                        <p style={{ fontFamily: "'Space Mono', monospace", fontSize: "0.62rem", color: "#334155", marginTop: 6 }}>
                            {cityName}
                        </p>
                    </div>
                </div>
            )}

            {/* ── ERROR ─────────────────────────────────────────────────────────── */}
            {error && !loading && (
                <div style={{
                    position: "absolute", bottom: 24, left: "50%", transform: "translateX(-50%)", zIndex: 50,
                    background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.3)",
                    borderRadius: 12, padding: "10px 18px", color: "#f87171", fontSize: "0.75rem",
                    fontFamily: "'Space Mono', monospace", display: "flex", alignItems: "center", gap: 8,
                    maxWidth: 500, textAlign: "center",
                }}>
                    <AlertTriangle size={13} style={{ flexShrink: 0 }} /> {error}
                </div>
            )}

            {/* Spin keyframe for loading button icon */}
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}
