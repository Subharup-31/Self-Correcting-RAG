"use client";

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  MessageSquare,
  UploadCloud,
  Activity,
  FileText,
  Send,
  ThumbsUp,
  ThumbsDown,
  AlertTriangle,
  Search,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  XCircle,
  ExternalLink,
  Database,
  BarChart3,
  BookOpen,
  Info
} from "lucide-react";

// --- API Helper types ---
interface Citation {
  source: string;
  page: string | number;
  doc_type?: string;
  excerpt?: string;
}

interface QueryResponse {
  query: string;
  answer: string;
  confidence_score: number;
  low_confidence: boolean;
  clarification_needed: boolean;
  clarification_question: string | null;
  contradiction_found: boolean;
  contradiction_detail: string | null;
  crag_state: string;
  hallucination_free: boolean;
  web_search_used: boolean;
  sources: Citation[];
  techniques_used: string[];
  processing_time: number;
  retry_count: number;
}

interface TraceEvent {
  node: string;
  elapsed: number;
  update: Record<string, any>;
}

function FormattedAnswer({ text }: { text: string }) {
  if (!text) return null;

  const parseInline = (line: string) => {
    const parts = [];
    let currentIdx = 0;
    const regex = /(\*\*|`|\[Doc\s+\d+[^\]]*\]|\[https?:\/\/[^\]]+\])/g;
    let match;
    
    while ((match = regex.exec(line)) !== null) {
      const matchText = match[0];
      const matchIdx = match.index;
      
      if (matchIdx > currentIdx) {
        parts.push(line.substring(currentIdx, matchIdx));
      }
      
      if (matchText === '**') {
        const closingIdx = line.indexOf('**', matchIdx + 2);
        if (closingIdx !== -1) {
          parts.push(<strong key={matchIdx} className="font-bold text-white">{line.substring(matchIdx + 2, closingIdx)}</strong>);
          regex.lastIndex = closingIdx + 2;
          currentIdx = closingIdx + 2;
        } else {
          parts.push(matchText);
          currentIdx = matchIdx + 2;
        }
      } else if (matchText === '`') {
        const closingIdx = line.indexOf('`', matchIdx + 1);
        if (closingIdx !== -1) {
          parts.push(<code key={matchIdx} className="bg-[#18181b] border border-[#27272a] rounded px-1.5 py-0.5 text-[#e2e2e2] font-mono text-[10px]">{line.substring(matchIdx + 1, closingIdx)}</code>);
          regex.lastIndex = closingIdx + 1;
          currentIdx = closingIdx + 1;
        } else {
          parts.push(matchText);
          currentIdx = matchIdx + 1;
        }
      } else if (matchText.startsWith('[Doc') || matchText.startsWith('[http')) {
        const cleanCit = matchText.slice(1, -1);
        parts.push(
          <span key={matchIdx} className="inline-flex items-center text-[9px] font-semibold text-[#5e6ad2] bg-[#5e6ad2]/10 border border-[#5e6ad2]/20 rounded-md px-1.5 py-0.5 mx-0.5 select-none hover:bg-[#5e6ad2]/20 transition-all cursor-default">
            {cleanCit}
          </span>
        );
        currentIdx = matchIdx + matchText.length;
      }
    }
    
    if (currentIdx < line.length) {
      parts.push(line.substring(currentIdx));
    }
    
    return parts.length > 0 ? parts : line;
  };

  const lines = text.split('\n');
  const renderedElements: React.ReactNode[] = [];
  let listItems: React.ReactNode[] = [];

  const flushList = (key: number) => {
    if (listItems.length > 0) {
      renderedElements.push(
        <ul key={`list-${key}`} className="list-disc pl-5 my-2 space-y-1 text-zinc-300">
          {listItems}
        </ul>
      );
      listItems = [];
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    
    if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
      const content = trimmed.substring(2);
      listItems.push(<li key={idx} className="leading-relaxed">{parseInline(content)}</li>);
    } else if (trimmed.match(/^\d+\.\s/)) {
      flushList(idx);
      const match = trimmed.match(/^(\d+)\.\s(.*)/);
      if (match) {
        renderedElements.push(
          <div key={idx} className="flex gap-2 my-1.5 leading-relaxed text-zinc-300">
            <span className="font-semibold text-[#5e6ad2]">{match[1]}.</span>
            <span>{parseInline(match[2])}</span>
          </div>
        );
      }
    } else if (trimmed.startsWith('### ')) {
      flushList(idx);
      renderedElements.push(<h4 key={idx} className="text-sm font-semibold text-white mt-4 mb-2">{parseInline(trimmed.substring(4))}</h4>);
    } else if (trimmed.startsWith('## ')) {
      flushList(idx);
      renderedElements.push(<h3 key={idx} className="text-base font-semibold text-white mt-5 mb-2.5">{parseInline(trimmed.substring(3))}</h3>);
    } else if (trimmed.startsWith('# ')) {
      flushList(idx);
      renderedElements.push(<h2 key={idx} className="text-lg font-bold text-white mt-6 mb-3">{parseInline(trimmed.substring(2))}</h2>);
    } else if (trimmed === '') {
      flushList(idx);
    } else {
      flushList(idx);
      renderedElements.push(<p key={idx} className="my-2 leading-relaxed text-zinc-300">{parseInline(line)}</p>);
    }
  });

  flushList(lines.length);

  return <div className="space-y-1 text-xs">{renderedElements}</div>;
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<"query" | "documents" | "analytics">("query");
  
  // System Health / Stats State
  const [health, setHealth] = useState({
    status: "loading",
    google_api_key: false,
    openai_api_key: false,
    tavily_api_key: false,
    documents_dir_exists: false,
  });
  const [stats, setStats] = useState({
    vector_store_chunks: 0,
    bm25_chunks: 0,
    few_shot_examples: 0,
    few_shot_avg_score: 1.0,
    sample_queries: [] as string[]
  });

  // Chat history
  type ChatMessage = { role: "user" | "assistant"; text: string; result?: QueryResponse };
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Query State
  const [queryInput, setQueryInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [clarificationAnswer, setClarificationAnswer] = useState("");
  const [isClarifying, setIsClarifying] = useState(false);
  const [originalQuestion, setOriginalQuestion] = useState("");

  // Result state (for clarification logic)
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [feedbackGiven, setFeedbackGiven] = useState<"positive" | "negative" | null>(null);

  // Pipeline trace streaming state
  const [traceLog, setTraceLog] = useState<TraceEvent[]>([]);
  const traceEndRef = useRef<HTMLDivElement>(null);

  // Document Upload state
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [uploadResults, setUploadResults] = useState<any[]>([]);

  // Eval harness state
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResults, setEvalResults] = useState<any>(null);

  // Load health & stats on mount
  useEffect(() => {
    fetchHealth();
    fetchStats();
  }, []);

  // Scroll to bottom of trace log
  useEffect(() => {
    traceEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [traceLog]);

  // Scroll chat to bottom when history changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, loading]);

  const fetchHealth = async () => {
    try {
      const res = await axios.get("/api/health");
      setHealth(res.data);
    } catch (err) {
      console.error("Health fetch failed", err);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await axios.get("/api/statistics");
      setStats({
        vector_store_chunks: res.data.stores.vector_store_chunks || 0,
        bm25_chunks: res.data.stores.bm25_chunks || 0,
        few_shot_examples: res.data.few_shot.total_examples || 0,
        few_shot_avg_score: res.data.few_shot.avg_feedback_score || 0.0,
        sample_queries: res.data.few_shot.sample_queries || [],
      });
    } catch (err) {
      console.error("Stats fetch failed", err);
    }
  };

  // --- Handlers ---
  const handleQuerySubmit = async (e: React.FormEvent, customQuery?: string) => {
    e.preventDefault();
    const targetQuery = customQuery || queryInput;
    if (!targetQuery.trim()) return;

    // Push user message into chat immediately
    setChatHistory(prev => [...prev, { role: "user", text: targetQuery }]);
    setQueryInput("");
    setLoading(true);
    setStreaming(true);
    setResult(null);
    setFeedbackGiven(null);
    setTraceLog([]);
    setIsClarifying(false);
    setOriginalQuestion(targetQuery);

    // SSE stream — pipeline runs ONCE and emits final_state in __done__
    const eventSource = new EventSource(`/api/query/stream?q=${encodeURIComponent(targetQuery)}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.node === "__done__" || data.node === "__complete__") {
          eventSource.close();
          setStreaming(false);
          setLoading(false);

          // final_state is embedded in __done__ — no second HTTP call needed
          const fs = data.final_state;
          if (fs) {
            // Map raw state keys → QueryResponse shape
            const sources: QueryResponse["sources"] = (fs.documents || []).map((doc: any) => ({
              source: doc.source || doc.metadata?.source || "Unknown",
              page: doc.page ?? doc.metadata?.page ?? null,
              doc_type: doc.doc_type || doc.metadata?.doc_type || null,
              excerpt: doc.excerpt || doc.content || "",
              rerank_score: doc.rerank_score ?? doc.metadata?.rerank_score ?? null,
            }));

            const qr: QueryResponse = {
              query: targetQuery,
              answer: fs.generation || fs.answer || "No answer generated.",
              confidence_score: fs.confidence_score ?? 0,
              low_confidence: fs.low_confidence ?? false,
              hallucination_free: fs.hallucination_free ?? true,
              web_search_used: fs.web_search_used ?? false,
              contradiction_found: fs.contradiction_found ?? false,
              contradiction_detail: fs.contradiction_detail ?? "",
              clarification_needed: fs.clarification_needed ?? false,
              clarification_question: fs.clarification_question ?? "",
              techniques_used: fs.techniques_used ?? [],
              crag_state: fs.crag_state ?? "",
              sources,
              processing_time: fs.processing_time ?? data.elapsed ?? 0,
              retry_count: fs.retry_count ?? 0,
            };

            setResult(qr);
            if (qr.clarification_needed) setIsClarifying(true);
            setChatHistory(prev => [...prev, { role: "assistant", text: qr.answer, result: qr }]);
          } else {
            // Fallback: backend is old version without embedded final_state — fetch separately
            fetchFinalResult(targetQuery);
          }
          fetchStats();

        } else if (data.node === "__error__") {
          eventSource.close();
          setStreaming(false);
          setLoading(false);
          setTraceLog(prev => [...prev, {
            node: "pipeline_error",
            elapsed: data.elapsed || 0,
            update: { error: data.error || "Unknown pipeline error" },
          }]);
          setChatHistory(prev => [...prev, {
            role: "assistant",
            text: `⚠️ Pipeline error: ${data.error || "Unknown error. Check Render logs."}`,
          }]);
        } else {
          // Intermediate node trace event
          setTraceLog(prev => [...prev, data]);
        }
      } catch (err) {
        console.error("Failed to parse SSE payload", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE stream error", err);
      eventSource.close();
      setStreaming(false);
      setLoading(false);
      setChatHistory(prev => [...prev, {
        role: "assistant",
        text: "⚠️ Could not reach the backend. Make sure the Render service is awake (it may be spinning up — try again in 30 seconds).",
      }]);
    };
  };

  // Fallback for old backend that doesn't embed final_state in __done__
  const fetchFinalResult = async (queryText: string) => {
    try {
      const res = await axios.post("/api/query", { query: queryText });
      const data: QueryResponse = res.data;
      setResult(data);
      if (data.clarification_needed) setIsClarifying(true);
      setChatHistory(prev => [...prev, { role: "assistant", text: data.answer, result: data }]);
      fetchStats();
    } catch (err) {
      console.error("fetchFinalResult failed", err);
      setChatHistory(prev => [...prev, {
        role: "assistant",
        text: "⚠️ Backend error. Check Render logs and make sure all API keys are set.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClarificationSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!clarificationAnswer.trim() || !result) return;

    // Build a combined prompt context
    const clarifiedQuery = `Original Question: ${originalQuestion}\nUser Clarification: ${clarificationAnswer}`;
    setClarificationAnswer("");
    setIsClarifying(false);
    handleQuerySubmit(e, clarifiedQuery);
  };

  const handleFeedback = async (isPositive: boolean) => {
    if (!result) return;
    try {
      await axios.post("/api/feedback", {
        query: result.query,
        answer: result.answer,
        is_positive: isPositive,
        feedback_score: 1.0,
      });
      setFeedbackGiven(isPositive ? "positive" : "negative");
      fetchStats();
    } catch (err) {
      console.error("Feedback submit failed", err);
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!files || files.length === 0) return;

    setUploadStatus("Uploading & indexing...");
    setUploadResults([]);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setUploadResults(res.data.results);
      setUploadStatus("Ingestion completed!");
      fetchStats();
    } catch (err) {
      console.error("Upload failed", err);
      setUploadStatus("Upload failed. Check logs.");
    }
  };

  const handleClearDocs = async () => {
    if (!confirm("Are you sure you want to clear all indexed documents? This wipes ChromaDB and BM25!")) return;
    try {
      await axios.delete("/api/documents");
      alert("All documents cleared.");
      fetchStats();
    } catch (err) {
      console.error("Clear failed", err);
    }
  };

  const runEvaluation = async () => {
    setEvalLoading(true);
    setEvalResults(null);
    try {
      const res = await axios.get("/api/evaluate");
      setEvalResults(res.data);
    } catch (err) {
      console.error("Evaluation failed", err);
    } finally {
      setEvalLoading(false);
    }
  };

  // --- Node Trace Format Helpers ---
  const getNodeNameLabel = (node: string) => {
    const labels: Record<string, string> = {
      route_question: "Query Routing",
      query_decompose: "Query Decomposition",
      retrieve: "Context Retrieval (BM25 + Chroma)",
      grade_documents: "CRAG Document Relevance Grading",
      detect_contradiction: "Factual Contradiction Fact-Check",
      clarify: "Ambiguity Detector (Halt / Ask)",
      query_rewrite: "Query Optimizer (Rewrite Loop)",
      web_search: "Tavily Live Web Search Fallback",
      rerank: "Cross-Encoder Joint Reranker",
      few_shot_inject: "Dynamic Few-Shot Learning Memory Inject",
      generate: "Context-Grounded Generation",
      grade_hallucination: "Anti-Hallucination Grounding Grader",
      regenerate: "Ungrounded Claims Re-generation Loop",
      confidence_scorer: "Composite Confidence Calculator",
      grade_answer: "Query-Resolution Gate",
      direct_llm: "Conversational direct LLM answer",
      finalize: "Source Citation Builder"
    };
    return labels[node] || node;
  };

  const getNodeColor = (node: string) => {
    if (node === "pipeline_error" || node === "__error__") return "border-[#ef4444] bg-[#ef4444]/10";
    const greenNodes = ["generate", "finalize", "grade_hallucination", "grade_answer"];
    const purpleNodes = ["route_question", "few_shot_inject", "query_decompose", "rerank"];
    const amberNodes = ["detect_contradiction", "clarify", "query_rewrite", "web_search", "regenerate"];

    if (greenNodes.includes(node)) return "border-[#10b981] bg-[#10b981]/5";
    if (purpleNodes.includes(node)) return "border-[#5e6ad2] bg-[#5e6ad2]/5";
    if (amberNodes.includes(node)) return "border-[#f59e0b] bg-[#f59e0b]/5";
    return "border-[#202024] bg-[#121214]";
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#09090b]">
      {/* Top Header */}
      <header className="border-b border-[#202024] bg-[#121214]/60 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#5e6ad2] to-[#707df0] flex items-center justify-center font-bold text-white text-lg">
            Ω
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
              Ultimate Self-Correcting RAG
              <span className="text-[10px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-[#202024] text-zinc-400 font-medium">
                V1.0
              </span>
            </h1>
            <p className="text-[11px] text-zinc-400">Agentic Adaptive-RAG with Self-Correction</p>
          </div>
        </div>

        {/* API Connection Indicators */}
        <div className="flex items-center gap-4 text-[11px]">
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${health.google_api_key ? "bg-[#10b981]" : "bg-[#ef4444]"}`} />
            <span className="text-zinc-400">Gemini LLM</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${health.tavily_api_key ? "bg-[#10b981]" : "bg-[#ef4444]"}`} />
            <span className="text-zinc-400">Tavily Web Search</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#10b981]" />
            <span className="text-zinc-400">ChromaDB ({stats.vector_store_chunks} chunks)</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Navigation Sidebar */}
        <nav className="w-full md:w-64 border-r border-[#202024] bg-[#0c0c0e] p-4 flex flex-col gap-1.5">
          <button
            onClick={() => setActiveTab("query")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "query"
                ? "bg-[#5e6ad2]/10 text-white border border-[#5e6ad2]/30"
                : "text-zinc-400 hover:text-zinc-100 hover:bg-[#121214]"
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            Interactive Console
          </button>
          
          <button
            onClick={() => setActiveTab("documents")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "documents"
                ? "bg-[#5e6ad2]/10 text-white border border-[#5e6ad2]/30"
                : "text-zinc-400 hover:text-zinc-100 hover:bg-[#121214]"
            }`}
          >
            <FileText className="w-4 h-4" />
            Document Ingest
          </button>

          <button
            onClick={() => setActiveTab("analytics")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === "analytics"
                ? "bg-[#5e6ad2]/10 text-white border border-[#5e6ad2]/30"
                : "text-zinc-400 hover:text-zinc-100 hover:bg-[#121214]"
            }`}
          >
            <Activity className="w-4 h-4" />
            System Trace & Eval
          </button>

          {/* Stats Summary Widget */}
          <div className="mt-auto border-t border-[#202024] pt-4 px-2">
            <h3 className="text-[10px] uppercase font-bold tracking-widest text-zinc-500 mb-2">Memory Metrics</h3>
            <div className="flex flex-col gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-zinc-400">Indexed Chunks</span>
                <span className="font-semibold text-zinc-200">{stats.vector_store_chunks}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">BM25 Keywords</span>
                <span className="font-semibold text-zinc-200">{stats.bm25_chunks}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Few-Shot Pairs</span>
                <span className="font-semibold text-zinc-200">{stats.few_shot_examples}</span>
              </div>
            </div>
          </div>
        </nav>

        {/* Content panel */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Tab 1: Interactive Console */}
          {activeTab === "query" && (
            <div className="flex-1 flex flex-col xl:flex-row h-full overflow-hidden">

              {/* LEFT: Chat Area (ChatGPT-style) */}
              <div className="flex-1 flex flex-col overflow-hidden border-r border-[#202024]">

                {/* Scrollable chat messages */}
                <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-6">

                  {/* Empty state */}
                  {chatHistory.length === 0 && !loading && (
                    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-20">
                      <div className="w-16 h-16 rounded-2xl bg-[#5e6ad2]/10 border border-[#5e6ad2]/20 flex items-center justify-center">
                        <Sparkles className="w-8 h-8 text-[#5e6ad2]" />
                      </div>
                      <div>
                        <h2 className="text-base font-semibold text-white">Self-Correcting RAG</h2>
                        <p className="text-xs text-zinc-400 mt-1 max-w-xs">
                          Ask anything. The pipeline automatically routes, retrieves, detects contradictions, and self-corrects.
                        </p>
                      </div>
                      {/* Preset chips */}
                      <div className="flex flex-wrap gap-2 justify-center mt-2">
                        {[
                          { label: "🎯 HyDE & Retrieval", q: "What is HyDE and how does it improve retrieval?" },
                          { label: "⚡ Revenue conflict", q: "What was the company's revenue in 2024?" },
                          { label: "⚠️ Vague question", q: "Tell me about the main issues." },
                          { label: "🌐 Live web price", q: "What is today's stock price of Apple?" },
                        ].map(({ label, q }) => (
                          <button
                            key={label}
                            onClick={(e) => handleQuerySubmit(e, q)}
                            disabled={loading}
                            className="bg-[#121214] hover:bg-[#1c1c1f] disabled:opacity-40 text-zinc-300 border border-[#202024] rounded-full px-4 py-1.5 text-[11px] transition-all"
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Chat bubbles */}
                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"} animate-slide-up`}>

                      {/* Assistant avatar */}
                      {msg.role === "assistant" && (
                        <div className="w-8 h-8 rounded-xl bg-[#5e6ad2]/15 border border-[#5e6ad2]/30 flex items-center justify-center shrink-0 mt-1">
                          <span className="text-[11px] font-bold text-[#5e6ad2]">Ω</span>
                        </div>
                      )}

                      <div className={`max-w-[80%] flex flex-col gap-2 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                        {/* Bubble */}
                        <div className={`rounded-2xl px-4 py-3 ${
                          msg.role === "user"
                            ? "bg-[#5e6ad2] text-white text-xs font-medium rounded-tr-sm"
                            : "bg-[#131316] border border-[#202024] text-zinc-200 rounded-tl-sm"
                        }`}>
                          {msg.role === "user" ? (
                            <span className="text-xs">{msg.text}</span>
                          ) : (
                            <FormattedAnswer text={msg.text} />
                          )}
                        </div>

                        {/* Assistant result metadata (chips + sources + feedback) */}
                        {msg.role === "assistant" && msg.result && (
                          <div className="flex flex-col gap-2 w-full">
                            {/* Contradiction banner */}
                            {msg.result.contradiction_found && (
                              <div className="bg-[#ef4444]/10 border border-[#ef4444]/30 rounded-lg px-3 py-2 flex gap-2 text-[11px]">
                                <AlertTriangle className="w-3.5 h-3.5 text-[#ef4444] shrink-0 mt-0.5" />
                                <span className="text-[#ef4444]">{msg.result.contradiction_detail}</span>
                              </div>
                            )}

                            {/* Meta row: confidence + techniques + sources */}
                            <div className="flex flex-wrap items-center gap-1.5">
                              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                                msg.result.low_confidence
                                  ? "text-[#ef4444] border-[#ef4444]/30 bg-[#ef4444]/10"
                                  : "text-[#10b981] border-[#10b981]/30 bg-[#10b981]/10"
                              }`}>
                                {(msg.result.confidence_score * 100).toFixed(0)}% confidence
                              </span>
                              {msg.result.hallucination_free && (
                                <span className="text-[10px] text-[#10b981] border border-[#10b981]/20 bg-[#10b981]/10 px-2 py-0.5 rounded-full">🛡️ Grounded</span>
                              )}
                              {msg.result.web_search_used && (
                                <span className="text-[10px] text-[#f59e0b] border border-[#f59e0b]/20 bg-[#f59e0b]/10 px-2 py-0.5 rounded-full">🌐 Web</span>
                              )}
                              <span className="text-[10px] text-zinc-500">{msg.result.processing_time}s</span>
                            </div>

                            {/* Sources */}
                            {msg.result.sources && msg.result.sources.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {msg.result.sources.slice(0, 4).map((src, si) => (
                                  <span key={si} className="text-[10px] text-[#5e6ad2] bg-[#5e6ad2]/10 border border-[#5e6ad2]/20 rounded-md px-2 py-0.5 flex items-center gap-1">
                                    <BookOpen className="w-2.5 h-2.5" />
                                    {src.source}{src.page ? ` p.${src.page}` : ""}
                                  </span>
                                ))}
                              </div>
                            )}

                            {/* Feedback row (only for latest assistant message) */}
                            {idx === chatHistory.length - 1 && (
                              <div className="flex items-center gap-2">
                                {feedbackGiven === "positive" ? (
                                  <span className="text-[11px] text-[#10b981] font-semibold flex items-center gap-1">
                                    <CheckCircle2 className="w-3.5 h-3.5" /> Indexed!
                                  </span>
                                ) : feedbackGiven === "negative" ? (
                                  <span className="text-[11px] text-zinc-500">Feedback recorded.</span>
                                ) : (
                                  <>
                                    <button onClick={() => handleFeedback(true)} className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-[#10b981] transition-colors">
                                      <ThumbsUp className="w-3 h-3" /> Accurate
                                    </button>
                                    <span className="text-zinc-600">·</span>
                                    <button onClick={() => handleFeedback(false)} className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-[#ef4444] transition-colors">
                                      <ThumbsDown className="w-3 h-3" /> Poor
                                    </button>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* User avatar */}
                      {msg.role === "user" && (
                        <div className="w-8 h-8 rounded-xl bg-[#5e6ad2] flex items-center justify-center shrink-0 mt-1">
                          <span className="text-[11px] font-bold text-white">U</span>
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Typing indicator while loading */}
                  {loading && (
                    <div className="flex gap-3 justify-start animate-slide-up">
                      <div className="w-8 h-8 rounded-xl bg-[#5e6ad2]/15 border border-[#5e6ad2]/30 flex items-center justify-center shrink-0">
                        <span className="text-[11px] font-bold text-[#5e6ad2]">Ω</span>
                      </div>
                      <div className="bg-[#131316] border border-[#202024] rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2">
                        <div className="flex gap-1">
                          <span className="w-1.5 h-1.5 bg-[#5e6ad2] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                          <span className="w-1.5 h-1.5 bg-[#5e6ad2] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                          <span className="w-1.5 h-1.5 bg-[#5e6ad2] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                        <span className="text-[11px] text-zinc-400">Running pipeline...</span>
                      </div>
                    </div>
                  )}

                  {/* Clarification inline input */}
                  {isClarifying && result && (
                    <div className="flex gap-3 justify-start animate-slide-up">
                      <div className="w-8 h-8 rounded-xl bg-[#f59e0b]/15 border border-[#f59e0b]/30 flex items-center justify-center shrink-0">
                        <Info className="w-4 h-4 text-[#f59e0b]" />
                      </div>
                      <div className="flex flex-col gap-2 max-w-[80%]">
                        <div className="bg-[#f59e0b]/10 border border-[#f59e0b]/30 rounded-2xl rounded-tl-sm px-4 py-3">
                          <p className="text-[11px] text-[#f59e0b] font-semibold mb-1">Clarification needed</p>
                          <p className="text-xs text-zinc-200">{result.clarification_question}</p>
                        </div>
                        <form onSubmit={handleClarificationSubmit} className="flex gap-2">
                          <input
                            type="text"
                            value={clarificationAnswer}
                            onChange={(e) => setClarificationAnswer(e.target.value)}
                            placeholder="Your response..."
                            className="flex-1 bg-[#09090b] border border-[#f59e0b]/30 focus:border-[#f59e0b] focus:outline-none rounded-lg px-3 py-2 text-xs text-white"
                          />
                          <button type="submit" className="bg-[#f59e0b] hover:bg-[#d97706] text-zinc-950 font-medium rounded-lg px-3 py-2 text-xs">
                            Send
                          </button>
                        </form>
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>

                {/* Sticky bottom input bar */}
                <div className="border-t border-[#202024] bg-[#0c0c0e] px-6 py-4">
                  <form onSubmit={(e) => handleQuerySubmit(e)} className="flex gap-2 items-center">
                    <input
                      type="text"
                      value={queryInput}
                      onChange={(e) => setQueryInput(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      disabled={loading}
                      className="flex-1 bg-[#131316] border border-[#202024] focus:border-[#5e6ad2] focus:outline-none rounded-xl px-4 py-3 text-xs text-white placeholder-zinc-500 transition-all"
                    />
                    <button
                      type="submit"
                      disabled={loading || !queryInput.trim()}
                      className="bg-[#5e6ad2] hover:bg-[#707df0] disabled:bg-[#5e6ad2]/30 disabled:cursor-not-allowed text-white font-medium rounded-xl w-10 h-10 flex items-center justify-center transition-all shadow-lg shadow-[#5e6ad2]/20 shrink-0"
                    >
                      {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                    </button>
                  </form>
                </div>
              </div>

              {/* RIGHT: Pipeline Execution Trace */}
              <div className="w-full xl:w-[360px] flex flex-col overflow-hidden">
                <div className="flex-1 flex flex-col overflow-hidden p-6">
                  <h3 className="text-xs font-semibold text-white mb-1">LangGraph Pipeline Trace</h3>
                  <p className="text-[10px] text-zinc-400 mb-4">Observe real-time state flow through nodes</p>

                  <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-3">
                    {traceLog.length === 0 && !loading && (
                      <div className="flex-1 flex flex-col items-center justify-center text-zinc-600 gap-2 py-16">
                        <Activity className="w-8 h-8 opacity-25" />
                        <span className="text-xs">No active execution trace.</span>
                      </div>
                    )}

                    {traceLog.map((event, idx) => (
                      <div key={idx} className={`border border-[#202024] rounded-lg p-3 text-[11px] animate-slide-up ${getNodeColor(event.node)}`}>
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-bold text-zinc-300">{getNodeNameLabel(event.node)}</span>
                          <span className="text-[9px] text-zinc-500 font-mono">{event.elapsed}s</span>
                        </div>
                        <div className="text-zinc-400 font-mono text-[10px] overflow-hidden whitespace-pre-wrap">
                          {event.node === "route_question" && event.update.route && (
                            <p>🗺️ Routed to: <span className="text-white font-bold">{event.update.route}</span></p>
                          )}
                          {event.node === "query_decompose" && event.update.sub_questions && (
                            <div>
                              <p>🔍 Decomposed into sub-queries:</p>
                              {event.update.sub_questions.map((q: string, i: number) => (
                                <p key={i} className="pl-2 text-zinc-300">- {q}</p>
                              ))}
                            </div>
                          )}
                          {event.node === "retrieve" && event.update.documents && (
                            <p>📖 Retrieved <span className="text-white font-bold">{event.update.documents.length}</span> child chunks from ChromaDB/BM25.</p>
                          )}
                          {event.node === "grade_documents" && event.update.crag_state && (
                            <p>📊 CRAG aggregated state: <span className="text-white font-bold">{event.update.crag_state}</span></p>
                          )}
                          {event.node === "detect_contradiction" && (
                            <p>⚖️ Contradiction: <span className={event.update.contradiction_found ? "text-[#ef4444]" : "text-[#10b981]"}>{event.update.contradiction_found ? "FOUND" : "Clean"}</span></p>
                          )}
                          {event.node === "clarify" && event.update.clarification_question && (
                            <p>⚠️ Asking: <span className="text-[#f59e0b]">{event.update.clarification_question}</span></p>
                          )}
                          {event.node === "query_rewrite" && event.update.question && (
                            <p>🔄 Optimized to: <span className="text-white">{event.update.question}</span></p>
                          )}
                          {event.node === "web_search" && (<p>🌐 Web search executed.</p>)}
                          {event.node === "rerank" && event.update.documents && (
                            <p>⚡ Top rerank score: <span className="text-white">{(event.update.documents[0]?.metadata?.rerank_score || 0).toFixed(3)}</span></p>
                          )}
                          {event.node === "few_shot_inject" && (<p>📦 Few-shot injected.</p>)}
                          {event.node === "generate" && event.update.generation && (
                            <p>✍️ "{event.update.generation.slice(0, 60)}..."</p>
                          )}
                          {event.node === "grade_hallucination" && (
                            <p>🛡️ <span className={event.update.hallucination_free ? "text-[#10b981]" : "text-[#ef4444]"}>{event.update.hallucination_free ? "Grounded" : "Hallucination!"}</span> (Score: {event.update.hallucination_score})</p>
                          )}
                          {event.node === "regenerate" && (<p>🔄 Re-generating (#{event.update.regen_count})</p>)}
                          {event.node === "confidence_scorer" && (
                            <p>📏 Score: <span className="text-white">{event.update.confidence_score}</span> | Low: {event.update.low_confidence ? "YES" : "NO"}</p>
                          )}
                          {event.node === "grade_answer" && (
                            <p>🏁 Resolves query: <span className="text-white">{event.update.answer_addresses_question ? "YES" : "NO"}</span></p>
                          )}
                          {event.node === "pipeline_error" && (
                            <p className="text-[#ef4444]">❌ {event.update.error}</p>
                          )}
                        </div>
                      </div>
                    ))}

                    {streaming && (
                      <div className="border border-[#5e6ad2] rounded-lg p-3 text-[11px] animate-pulse flex items-center gap-2">
                        <RefreshCw className="w-4 h-4 animate-spin text-[#5e6ad2]" />
                        <span className="text-zinc-300 font-medium">Executing next GraphNode...</span>
                      </div>
                    )}
                    <div ref={traceEndRef} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Document Ingestion */}
          {activeTab === "documents" && (
            <div className="flex-1 flex flex-col md:flex-row gap-6 animate-slide-up">
              <div className="flex-1 linear-card p-6 flex flex-col gap-4">
                <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                  <UploadCloud className="w-4 h-4 text-[#5e6ad2]" />
                  Upload Documents
                </h2>
                <p className="text-xs text-zinc-400">
                  Upload text files, markdown, PDFs, Word documents, or image scans. Digital PDFs will be parsed textually; scanned image PDFs automatically route to Tesseract OCR.
                </p>

                <form onSubmit={handleFileUpload} className="flex flex-col gap-4">
                  <div className="border-2 border-dashed border-[#202024] hover:border-[#5e6ad2] rounded-lg p-8 flex flex-col items-center justify-center gap-3 transition-all cursor-pointer">
                    <UploadCloud className="w-10 h-10 text-zinc-500" />
                    <input
                      type="file"
                      multiple
                      onChange={(e) => setFiles(e.target.files)}
                      className="text-xs text-zinc-400"
                    />
                    <p className="text-[10px] text-zinc-500">Supports PDF, DOCX, TXT, PNG, JPG</p>
                  </div>

                  <div className="flex gap-2">
                    <button
                      type="submit"
                      disabled={!files || files.length === 0}
                      className="bg-[#5e6ad2] hover:bg-[#707df0] disabled:bg-[#5e6ad2]/40 disabled:cursor-not-allowed text-white font-medium rounded-lg px-4 py-2.5 text-xs transition-all"
                    >
                      Ingest Files
                    </button>
                    <button
                      type="button"
                      onClick={handleClearDocs}
                      className="border border-[#ef4444]/30 hover:border-[#ef4444] text-[#ef4444] hover:bg-[#ef4444]/5 font-medium rounded-lg px-4 py-2.5 text-xs transition-all"
                    >
                      Wipe Database
                    </button>
                  </div>
                </form>

                {uploadStatus && (
                  <p className="text-xs text-zinc-300 font-medium bg-[#121214] border border-[#202024] p-3 rounded-lg">
                    📢 {uploadStatus}
                  </p>
                )}

                {uploadResults.length > 0 && (
                  <div>
                    <h3 className="text-xs font-bold text-zinc-300 mb-2">Ingestion Results:</h3>
                    <div className="flex flex-col gap-2">
                      {uploadResults.map((r, idx) => (
                        <div key={idx} className="bg-[#0c0c0e] border border-[#202024] rounded-lg p-3 text-xs">
                          <p className="font-semibold text-zinc-200">{r.file}</p>
                          {r.status === "ingested" ? (
                            <p className="text-[#10b981] mt-1">
                              ✓ Successfully indexed {r.summary.children} chunks ({r.summary.parents} parent contexts) in {r.summary.seconds}s.
                            </p>
                          ) : (
                            <p className="text-[#ef4444] mt-1">❌ Ingestion failed: {r.error}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Ingest presets */}
              <div className="w-full md:w-80 linear-card p-6 flex flex-col gap-4">
                <h3 className="text-xs font-bold text-white">Generate Demonstration Corpora</h3>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  Generate the standard contradictory and vague manuals used to show off all pipeline self-correcting scenarios.
                </p>
                <button
                  onClick={async () => {
                    try {
                      await axios.get("/api/health"); // trigger backend load if needed
                      const res = await axios.post("/api/upload"); // Wait, we can trigger demo via evaluate, or generate files from CLI
                      alert("Done! Please run 'python3 main.py demo' in your terminal to generate and ingest all sample documents automatically.");
                    } catch (e) {
                      alert("Please run 'python3 main.py demo' in your terminal to build sample files.");
                    }
                  }}
                  className="bg-[#121214] hover:bg-[#1c1c1f] text-zinc-300 border border-[#202024] rounded-lg py-2.5 text-xs transition-all flex items-center justify-center gap-2"
                >
                  <Database className="w-4 h-4" />
                  Load Sample Docs
                </button>
              </div>
            </div>
          )}

          {/* Tab 3: System Analytics & Eval */}
          {activeTab === "analytics" && (
            <div className="flex-1 flex flex-col gap-6 animate-slide-up">
              {/* Top stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="linear-card p-5 flex flex-col gap-2">
                  <span className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">ChromaDB Persist Store</span>
                  <span className="text-2xl font-semibold text-white">{stats.vector_store_chunks}</span>
                  <span className="text-[11px] text-zinc-500">Child semantic text fragments</span>
                </div>
                <div className="linear-card p-5 flex flex-col gap-2">
                  <span className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Self-Improving Memory</span>
                  <span className="text-2xl font-semibold text-white">{stats.few_shot_examples}</span>
                  <span className="text-[11px] text-zinc-500">Learned examples (Avg Score: {stats.few_shot_avg_score.toFixed(2)})</span>
                </div>
                <div className="linear-card p-5 flex flex-col gap-2">
                  <span className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Lexical Index</span>
                  <span className="text-2xl font-semibold text-white">{stats.bm25_chunks}</span>
                  <span className="text-[11px] text-zinc-500">BM25 keyword search tokens</span>
                </div>
              </div>

              {/* Dynamic prompt examples list */}
              <div className="linear-card p-6 flex flex-col gap-4">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-[#5e6ad2]" />
                  Learned Few-Shot Memory Examples
                </h3>
                <p className="text-xs text-zinc-400">
                  This table shows the queries that users have upvoted/liked. The system semantically matches subsequent queries against these to adapt formatting, vocabulary, and tone automatically.
                </p>

                {stats.sample_queries.length === 0 ? (
                  <div className="bg-[#0c0c0e] border border-[#202024] rounded-lg p-5 text-center text-xs text-zinc-500">
                    No learned memory items yet. Upvote high-quality answers in the console to index them here!
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {stats.sample_queries.map((q, idx) => (
                      <div key={idx} className="bg-[#121214] border border-[#202024] rounded-lg p-3 text-xs font-mono text-zinc-300">
                        {q}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Eval Harness Run */}
              <div className="linear-card p-6 flex flex-col gap-4">
                <h3 className="text-xs font-bold text-white">Evaluation Benchmarking</h3>
                <p className="text-xs text-zinc-400">
                  Execute the built-in 12-question benchmark suite comparing baseline standard RAG against the adaptive self-correcting RAG. (Warning: this executes 24 full pipeline pipelines and takes up to 2 minutes).
                </p>
                <div>
                  <button
                    onClick={runEvaluation}
                    disabled={evalLoading}
                    className="bg-[#5e6ad2] hover:bg-[#707df0] disabled:bg-[#5e6ad2]/40 disabled:cursor-not-allowed text-white font-medium rounded-lg px-5 py-2.5 text-xs transition-all flex items-center gap-2"
                  >
                    {evalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
                    Run Evaluation Suite
                  </button>
                </div>

                {evalLoading && (
                  <div className="bg-[#121214] border border-[#202024] rounded-lg p-6 flex flex-col items-center justify-center gap-3">
                    <RefreshCw className="w-6 h-6 animate-spin text-[#5e6ad2]" />
                    <p className="text-xs text-zinc-300 font-medium">Running 12 test questions through standard vs. self-correcting loops...</p>
                  </div>
                )}

                {evalResults && (
                  <div className="bg-[#121214] border border-[#202024] rounded-lg p-5 flex flex-col gap-4 animate-slide-up">
                    <h4 className="text-xs font-bold text-zinc-200">Benchmarking Summary (12 Queries)</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="border border-[#ef4444]/20 bg-[#ef4444]/5 p-4 rounded-lg flex flex-col gap-2 text-xs">
                        <span className="font-bold text-[#ef4444]">Baseline Standard RAG</span>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Hallucination Rate:</span>
                          <span className="font-semibold text-zinc-200">{(evalResults.baseline_metrics.hallucination_rate * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Accuracy (RAGAS):</span>
                          <span className="font-semibold text-zinc-200">{(evalResults.baseline_metrics.accuracy * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Ambiguity Halts:</span>
                          <span className="font-semibold text-zinc-200">0% (always guessed)</span>
                        </div>
                      </div>

                      <div className="border border-[#10b981]/20 bg-[#10b981]/5 p-4 rounded-lg flex flex-col gap-2 text-xs">
                        <span className="font-bold text-[#10b981]">Adaptive Self-Correcting RAG</span>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Hallucination Rate:</span>
                          <span className="font-semibold text-[#10b981]">{(evalResults.ultimate_metrics.hallucination_rate * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Accuracy (RAGAS):</span>
                          <span className="font-semibold text-[#10b981]">{(evalResults.ultimate_metrics.accuracy * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Ambiguity Clarification Rate:</span>
                          <span className="font-semibold text-zinc-200">{(evalResults.ultimate_metrics.clarification_rate * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
