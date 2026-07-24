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
  Info,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Compass,
  Scale,
  Globe,
  Zap,
  HelpCircle,
  Target,
  ShieldAlert,
  ShieldCheck,
  Package,
  Flag
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
  clarification_options?: string[] | null;
  contradiction_found: boolean;
  contradiction_detail: string | null;
  crag_state: string;
  hallucination_free: boolean;
  web_search_used: boolean;
  sources: Citation[];
  techniques_used: string[];
  processing_time: number;
  retry_count: number;
  confidence_reason?: string;
}

interface TraceEvent {
  node: string;
  elapsed: number;
  update: Record<string, any>;
}

// ============================================================
// AssistantMessage — Claude-style chat bubble with toolbar
// ============================================================
interface AssistantMessageProps {
  msg: { 
    role: "user" | "assistant"; 
    text: string; 
    result?: any; 
    clarifiedValue?: string;
  };
  idx: number;
  isLatest: boolean;
  feedbackGiven: "positive" | "negative" | null;
  onFeedback: (positive: boolean) => void;
  onClarifySelect: (value: string) => void;
  showTextInput: boolean;
  setShowTextInput: (show: boolean) => void;
  clarificationAnswer: string;
  setClarificationAnswer: (val: string) => void;
  loading: boolean;
}

function AssistantMessage({ 
  msg, 
  idx, 
  isLatest, 
  feedbackGiven, 
  onFeedback,
  onClarifySelect,
  showTextInput,
  setShowTextInput,
  clarificationAnswer,
  setClarificationAnswer,
  loading
}: AssistantMessageProps) {
  const [copied, setCopied] = React.useState(false);

  const copyText = () => {
    navigator.clipboard.writeText(msg.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex gap-3 justify-start animate-slide-up">
      {/* Ω Avatar */}
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 mt-0.5 ${msg.result?.clarification_needed && !msg.clarifiedValue ? 'bg-[#f59e0b]/15 border border-[#f59e0b]/30 text-[#f59e0b]' : 'bg-[#5e6ad2]/15 border border-[#5e6ad2]/30 text-[#5e6ad2]'}`}>
        <span className="text-[11px] font-bold">Ω</span>
      </div>

      <div className="flex-1 flex flex-col gap-2 min-w-0 pb-2">
        {/* Contradiction banner */}
        {msg.result?.contradiction_found && (
          <div className="bg-[#ef4444]/10 border border-[#ef4444]/30 rounded-lg px-3 py-2 flex gap-2 text-[11px]">
            <AlertTriangle className="w-3.5 h-3.5 text-[#ef4444] shrink-0 mt-0.5" />
            <span className="text-[#ef4444]">{msg.result.contradiction_detail}</span>
          </div>
        )}

        {/* Answer text — clean, no box, like Claude */}
        <div className="text-zinc-200 text-base leading-[1.5]">
          <FormattedAnswer text={msg.text} />
        </div>

        {/* Clarified summary box */}
        {msg.result?.clarification_needed && msg.clarifiedValue && (
          <div className="bg-[#1c1c1f]/40 border border-[#2e2e33]/50 rounded-xl px-4 py-3 max-w-md my-1.5 text-sm">
            <div className="font-semibold text-zinc-300 mb-1">{msg.result.clarification_question}</div>
            <div className="text-[#5e6ad2] font-medium">{msg.clarifiedValue}</div>
          </div>
        )}

        {/* Clarification Multiple Choice Options (Vibe pattern) inline */}
        {msg.result?.clarification_needed && isLatest && !msg.clarifiedValue && (
          <div className="flex flex-col gap-3 max-w-[85%] bg-[#121214] border border-[#202024] rounded-2xl p-5 shadow-xl mt-2">
            {/* Title and Header */}
            <div className="flex justify-between items-start gap-4">
              <h3 className="text-sm font-semibold text-zinc-100 leading-normal">
                {msg.result.clarification_question || "Please clarify your request:"}
              </h3>
              <button
                onClick={() => onClarifySelect("Answer generally based on available information")}
                className="text-zinc-500 hover:text-zinc-300 transition-colors p-0.5"
                title="Skip"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>

            {!showTextInput ? (
              <div className="flex flex-col gap-2">
                {/* Options generated dynamically */}
                {msg.result.clarification_options && msg.result.clarification_options.map((opt: string, idx: number) => (
                  <button
                    key={idx}
                    onClick={() => onClarifySelect(opt)}
                    className="w-full text-left bg-[#1c1c1f]/40 hover:bg-[#202024] border border-[#2e2e33]/50 hover:border-zinc-700 rounded-xl px-4 py-3 text-zinc-200 hover:text-white transition-all flex items-center gap-3 text-sm font-medium"
                  >
                    <span className="flex items-center justify-center w-5 h-5 rounded bg-[#09090b] border border-[#2e2e33] text-[10px] font-bold text-zinc-400">
                      {idx + 1}
                    </span>
                    <span className="flex-1 truncate">{opt}</span>
                  </button>
                ))}

                {/* "Something else" Option */}
                <button
                  onClick={() => setShowTextInput(true)}
                  className="w-full text-left bg-[#1c1c1f]/40 hover:bg-[#202024] border border-[#2e2e33]/50 hover:border-zinc-700 rounded-xl px-4 py-3 text-zinc-200 hover:text-white transition-all flex items-center gap-3 text-sm font-medium"
                >
                  <span className="flex items-center justify-center w-5 h-5 rounded bg-[#09090b] border border-[#2e2e33] text-zinc-400">
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 20h9"/>
                      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
                    </svg>
                  </span>
                  <span>Something else</span>
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <form 
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (clarificationAnswer.trim()) {
                      onClarifySelect(clarificationAnswer);
                    }
                  }} 
                  className="flex gap-2"
                >
                  <input
                    type="text"
                    value={clarificationAnswer}
                    onChange={(e) => setClarificationAnswer(e.target.value)}
                    placeholder="Tell me more specifically..."
                    autoFocus
                    className="flex-1 bg-[#09090b] border border-zinc-700 focus:border-[#5e6ad2] focus:outline-none rounded-lg px-3 py-2.5 text-sm text-white"
                  />
                  <button type="submit" className="bg-[#5e6ad2] hover:bg-[#4b55be] text-white font-medium rounded-lg px-4 py-2.5 text-sm transition-all">
                    Send
                  </button>
                </form>
                
                <button
                  onClick={() => setShowTextInput(false)}
                  className="text-xs text-zinc-500 hover:text-zinc-300 font-medium flex items-center gap-1.5 transition-colors self-start"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="19" y1="12" x2="5" y2="12" />
                    <polyline points="12 19 5 12 12 5" />
                  </svg>
                  Back to options
                </button>
              </div>
            )}

            <div className="flex justify-between items-center border-t border-[#202024]/60 pt-3">
              <span className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">Clarification Option</span>
              <button
                onClick={() => onClarifySelect("Answer generally based on available information")}
                className="bg-[#1c1c1f]/40 hover:bg-[#202024] border border-[#2e2e33]/50 rounded-lg px-3 py-1.5 text-[11px] text-zinc-400 hover:text-zinc-200 font-semibold transition-all"
              >
                Skip
              </button>
            </div>
          </div>
        )}

        {/* Source chips */}
        {msg.result?.sources && msg.result.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1">
            {msg.result.sources.slice(0, 5).map((src: any, si: number) => (
              <span key={si} className="text-[10px] text-[#5e6ad2] bg-[#5e6ad2]/10 border border-[#5e6ad2]/20 rounded-md px-2 py-0.5 flex items-center gap-1">
                <BookOpen className="w-2.5 h-2.5" />
                {src.source}{src.page ? ` (Pages: ${src.page})` : ""}
              </span>
            ))}
          </div>
        )}

        {/* Toolbar row — Copy + Thumbs + metadata */}
        {!(msg.result?.clarification_needed && !msg.clarifiedValue) && (
          <div className="flex items-center gap-0.5 mt-0.5">
            <button onClick={copyText} title="Copy response" className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-all">
              {copied ? <Check className="w-3.5 h-3.5 text-[#10b981]" /> : <Copy className="w-3.5 h-3.5" />}
            </button>

            {isLatest && (
              <>
                <div className="w-px h-3 bg-zinc-700 mx-1" />
                {feedbackGiven === "positive" ? (
                  <span className="flex items-center gap-1 text-[10px] text-[#10b981] px-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Indexed!
                  </span>
                ) : feedbackGiven === "negative" ? (
                  <span className="text-[10px] text-zinc-500 px-1">Feedback noted.</span>
                ) : (
                  <>
                    <button onClick={() => onFeedback(true)} title="Good response" className="p-1.5 rounded-md text-zinc-500 hover:text-[#10b981] hover:bg-[#10b981]/5 transition-all">
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={() => onFeedback(false)} title="Poor response" className="p-1.5 rounded-md text-zinc-500 hover:text-[#ef4444] hover:bg-[#ef4444]/5 transition-all">
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </>
            )}

            {/* Quality tags (Confidence / Latency / Websearch) */}
            {msg.result && (
              <>
                <div className="w-px h-3 bg-zinc-700 mx-1" />
                <div className="flex items-center gap-1.5 text-[10px] text-zinc-500 font-semibold px-2">
                  {msg.result.web_search_used && (
                    <span className="text-zinc-400 bg-zinc-800 border border-zinc-700 rounded-md px-1.5 py-0.5">
                      WEB
                    </span>
                  )}
                  {msg.result.low_confidence && (
                    <span className="text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-md px-1.5 py-0.5 flex items-center gap-1" title={msg.result.confidence_reason}>
                      <ShieldAlert className="w-3 h-3 text-[#ef4444]" /> Low Confidence
                    </span>
                  )}
                  <span>
                    {Math.round(msg.result.confidence_score * 100)}%
                  </span>
                  <span>•</span>
                  <span>
                    {msg.result.processing_time.toFixed(3)}s
                  </span>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function FormattedAnswer({ text }: { text: string }) {
  if (!text) return null;

  const parseInline = (line: string) => {
    const parts = [];
    let currentIdx = 0;
    const regex = /(\*\*|`|\[[^\]]+\.[a-zA-Z0-9]{2,4}\s*,\s*pg\s*-\s*[^\]]+\]|\[Doc\s+\d+[^\]]*\]|\[https?:\/\/[^\]]+\])/g;
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
          parts.push(<code key={matchIdx} className="bg-[#18181b] border border-[#27272a] rounded px-1.5 py-0.5 text-[#e2e2e2] font-mono text-xs">{line.substring(matchIdx + 1, closingIdx)}</code>);
          regex.lastIndex = closingIdx + 1;
          currentIdx = closingIdx + 1;
        } else {
          parts.push(matchText);
          currentIdx = matchIdx + 1;
        }
      } else if (matchText.startsWith('[')) {
        const cleanCit = matchText.slice(1, -1);
        parts.push(
          <span key={matchIdx} className="inline-flex items-center text-[11px] font-semibold text-[#5e6ad2] bg-[#5e6ad2]/10 border border-[#5e6ad2]/20 rounded-md px-1.5 py-0.5 mx-0.5 select-none hover:bg-[#5e6ad2]/20 transition-all cursor-default">
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
      listItems.push(<li key={idx} className="leading-[1.5]">{parseInline(content)}</li>);
    } else if (trimmed.match(/^\d+\.\s/)) {
      flushList(idx);
      const match = trimmed.match(/^(\d+)\.\s(.*)/);
      if (match) {
        renderedElements.push(
          <div key={idx} className="flex gap-2 my-1.5 leading-[1.5] text-zinc-300">
            <span className="font-semibold text-[#5e6ad2]">{match[1]}.</span>
            <span>{parseInline(match[2])}</span>
          </div>
        );
      }
    } else if (trimmed.startsWith('### ')) {
      flushList(idx);
      renderedElements.push(<h4 key={idx} className="text-base font-semibold text-white mt-4 mb-2">{parseInline(trimmed.substring(4))}</h4>);
    } else if (trimmed.startsWith('## ')) {
      flushList(idx);
      renderedElements.push(<h3 key={idx} className="text-lg font-semibold text-white mt-5 mb-2.5">{parseInline(trimmed.substring(3))}</h3>);
    } else if (trimmed.startsWith('# ')) {
      flushList(idx);
      renderedElements.push(<h2 key={idx} className="text-xl font-bold text-white mt-6 mb-3">{parseInline(trimmed.substring(2))}</h2>);
    } else if (trimmed === '') {
      flushList(idx);
    } else {
      flushList(idx);
      renderedElements.push(<p key={idx} className="my-2 leading-[1.5] text-zinc-300">{parseInline(line)}</p>);
    }
  });

  flushList(lines.length);

  return <div className="space-y-2 text-base leading-[1.5]">{renderedElements}</div>;
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<"query" | "documents" | "analytics">("query");
  const [traceCollapsed, setTraceCollapsed] = useState(false);
  const [keepPersistent, setKeepPersistent] = useState<boolean>(true);
  const [sessionId] = useState(() => {
    if (typeof window !== "undefined") {
      let id = sessionStorage.getItem("rag_session_id");
      if (!id) {
        id = "sess_" + Math.random().toString(36).substring(2, 15);
        sessionStorage.setItem("rag_session_id", id);
      }
      return id;
    }
    return "";
  });
  
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
  type ChatMessage = { 
    role: "user" | "assistant"; 
    text: string; 
    result?: QueryResponse;
    clarifiedValue?: string;
  };
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Query State
  const [queryInput, setQueryInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [clarificationAnswer, setClarificationAnswer] = useState("");
  const [isClarifying, setIsClarifying] = useState(false);
  const [showTextInput, setShowTextInput] = useState(false);
  const [originalQuestion, setOriginalQuestion] = useState("");

  // Result state (for clarification logic)
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [feedbackGiven, setFeedbackGiven] = useState<"positive" | "negative" | null>(null);

  // Pipeline trace streaming state
  const [traceLog, setTraceLog] = useState<TraceEvent[]>([]);
  const traceEndRef = useRef<HTMLDivElement>(null);
  // Per-node LLM telemetry from final_state (provider, latency, cache hits)
  const [nodeTelemetry, setNodeTelemetry] = useState<any[]>([]);
  const [totalLlmCalls, setTotalLlmCalls] = useState<number>(0);

  // Document Upload state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [isDragging, setIsDragging] = useState(false);
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
  const handleQuerySubmit = async (e: React.FormEvent, customQuery?: string, isClarifyingSubmit = false) => {
    e.preventDefault();
    const targetQuery = customQuery || queryInput;
    if (!targetQuery.trim()) return;

    if (!isClarifyingSubmit) {
      setChatHistory(prev => [...prev, { role: "user", text: targetQuery }]);
    }
    setQueryInput("");
    setLoading(true);
    setStreaming(true);
    setResult(null);
    setFeedbackGiven(null);
    setTraceLog([]);
    setNodeTelemetry([]);
    setTotalLlmCalls(0);
    setIsClarifying(false);
    setOriginalQuestion(targetQuery);

    // SSE stream — pipeline runs ONCE and emits final_state in __done__
    const eventSource = new EventSource(`/api/query/stream?q=${encodeURIComponent(targetQuery)}&session_id=${sessionId}`);

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
            const sources: QueryResponse["sources"] = fs.sources || [];

            const qr: QueryResponse = {
              query: targetQuery,
              answer: fs.clarification_needed ? "Opted to solicit clarification before proceeding" : (fs.generation || fs.answer || "No answer generated."),
              confidence_score: fs.confidence_score ?? 0,
              low_confidence: fs.low_confidence ?? false,
              hallucination_free: fs.hallucination_free ?? true,
              web_search_used: fs.web_search_used ?? false,
              contradiction_found: fs.contradiction_found ?? false,
              contradiction_detail: fs.contradiction_detail ?? "",
              clarification_needed: fs.clarification_needed ?? false,
              clarification_question: fs.clarification_question ?? "",
              clarification_options: fs.clarification_options ?? [],
              techniques_used: fs.techniques_used ?? [],
              crag_state: fs.crag_state ?? "",
              sources,
              processing_time: fs.processing_time ?? data.elapsed ?? 0,
              retry_count: fs.retry_count ?? 0,
            };

            setResult(qr);
            // Surface per-node LLM telemetry from final_state
            if (fs.node_telemetry && Array.isArray(fs.node_telemetry)) {
              setNodeTelemetry(fs.node_telemetry);
            } else {
              setNodeTelemetry([]);
            }
            setTotalLlmCalls(fs.llm_calls ?? 0);
            if (qr.clarification_needed) {
              setIsClarifying(true);
              setShowTextInput(false);
            }
            setChatHistory(prev => {
              const next = [...prev];
              let found = false;
              for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].role === "assistant" && next[i].result?.clarification_needed && next[i].clarifiedValue) {
                  next[i].text = qr.answer;
                  next[i].result = qr;
                  found = true;
                  break;
                }
              }
              if (!found) {
                next.push({ role: "assistant", text: qr.answer, result: qr });
              }
              return next;
            });
          } else {
            // Fallback: backend is old version without embedded final_state — fetch separately
            fetchFinalResult(targetQuery, isClarifyingSubmit);
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
            text: `Pipeline error: ${data.error || "Unknown error. Check Render logs."}`,
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
        text: "Could not reach the backend. Make sure the Render service is awake (it may be spinning up — try again in 30 seconds).",
      }]);
    };
  };

  // Fallback for old backend that doesn't embed final_state in __done__
  const fetchFinalResult = async (queryText: string, isClarifyingSubmit = false) => {
    try {
      const res = await axios.post("/api/query", { query: queryText, session_id: sessionId });
      const data: QueryResponse = res.data;
      setResult(data);
      if (data.clarification_needed) {
        setIsClarifying(true);
        setShowTextInput(false);
      }
      setChatHistory(prev => {
        const next = [...prev];
        let found = false;
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "assistant" && next[i].result?.clarification_needed && next[i].clarifiedValue) {
            next[i].text = data.answer;
            next[i].result = data;
            found = true;
            break;
          }
        }
        if (!found) {
          next.push({ role: "assistant", text: data.answer, result: data });
        }
        return next;
      });
      fetchStats();
    } catch (err) {
      console.error("fetchFinalResult failed", err);
      setChatHistory(prev => [...prev, {
        role: "assistant",
        text: "Backend error. Check Render logs and make sure all API keys are set.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClarificationSelect = (value: string) => {
    if (!result) return;
    
    // Set clarifiedValue in chatHistory immediately
    setChatHistory(prev => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant" && next[i].result?.clarification_needed) {
          next[i].clarifiedValue = value;
          break;
        }
      }
      return next;
    });

    const clarifiedQuery = `Original Question: ${originalQuestion}\nUser Clarification: ${value}`;
    setClarificationAnswer("");
    setIsClarifying(false);
    setShowTextInput(false);
    const dummyEvent = { preventDefault: () => {} } as React.FormEvent;
    handleQuerySubmit(dummyEvent, clarifiedQuery, true);
  };

  const handleClarificationSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!clarificationAnswer.trim() || !result) return;
    handleClarificationSelect(clarificationAnswer);
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(e.dataTransfer.files);
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
    formData.append("session_id", sessionId);
    formData.append("persistent", keepPersistent ? "true" : "false");

    try {
      const res = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setUploadResults(res.data.results);
      setUploadStatus("Ingestion completed!");
      setFiles(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      fetchStats();
    } catch (err) {
      console.error("Upload failed", err);
      setUploadStatus("Upload failed. Check logs.");
    }
  };

  const handleClearDocs = async () => {
    if (!confirm("Are you sure you want to clear all indexed documents? This wipes ChromaDB, Qdrant, and BM25!")) return;
    try {
      await axios.delete("/api/documents");
      alert("All documents cleared.");
      fetchStats();
    } catch (err) {
      console.error("Clear failed", err);
    }
  };

  const handleClearFewShot = async () => {
    if (!confirm("Are you sure you want to clear all learned few-shot examples? This resets the AI's reinforcement memory!")) return;
    try {
      await axios.delete("/api/fewshot");
      alert("AI learned memory reset.");
      fetchStats();
    } catch (err) {
      console.error("Clear memory failed", err);
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
    <div className="h-screen max-h-screen flex flex-col bg-[#09090b] overflow-hidden">
      {/* Top Header */}
      <header className="border-b border-[#202024] bg-[#121214]/60 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#5e6ad2] to-[#707df0] flex items-center justify-center font-bold text-white text-lg">
            Ω
          </div>
          <div>
            <h1 className="font-instrument text-2xl font-bold text-white flex items-center gap-2">
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

      {/* Rate Limit Disclaimer Banner */}
      <div className="border-b border-[#f59e0b]/15 bg-[#f59e0b]/5 px-6 py-2.5 text-[11px] text-[#f59e0b]/90 flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-[#f59e0b] flex-shrink-0" />
        <span>
          <strong>Rate Limit Notice:</strong> This platform utilizes free/shared API tiers (e.g. Gemini API, Tavily, etc.) for demonstration purposes. Rapid operations, batch document uploads, or concurrent querying may occasionally trigger rate limit errors.
        </span>
      </div>

      {/* Main Grid */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Navigation Sidebar */}
        <nav className="w-full md:w-64 border-r border-[#202024] bg-[#0c0c0e] p-4 flex flex-col gap-1.5 overflow-y-auto shrink-0">
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
                        <h2 className="font-instrument text-3xl font-bold text-white">Self-Correcting RAG</h2>
                        <p className="text-xs text-zinc-400 mt-1 max-w-xs">
                          Ask anything. The pipeline automatically routes, retrieves, detects contradictions, and self-corrects.
                        </p>
                      </div>
                      {/* Preset chips */}
                      <div className="flex flex-wrap gap-2 justify-center mt-2">
                        {[
                          { label: "HyDE & Retrieval", icon: <Target className="w-3.5 h-3.5 text-[#5e6ad2]" />, q: "What is HyDE and how does it improve retrieval?" },
                          { label: "Revenue conflict", icon: <Zap className="w-3.5 h-3.5 text-[#f59e0b]" />, q: "What was the company's revenue in 2024?" },
                          { label: "Vague question", icon: <HelpCircle className="w-3.5 h-3.5 text-[#ef4444]" />, q: "Tell me about the main issues." },
                          { label: "Live web price", icon: <Globe className="w-3.5 h-3.5 text-[#10b981]" />, q: "What is today's stock price of Apple?" },
                        ].map(({ label, icon, q }) => (
                          <button
                            key={label}
                            onClick={(e) => handleQuerySubmit(e, q)}
                            disabled={loading}
                            className="flex items-center gap-1.5 bg-[#121214] hover:bg-[#1c1c1f] disabled:opacity-40 text-zinc-300 border border-[#202024] rounded-full px-4 py-1.5 text-[13px] transition-all"
                          >
                            {icon}
                            <span>{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Chat messages */}
                  {chatHistory.map((msg, idx) => {
                    const isLatest = idx === chatHistory.length - 1;
                    if (msg.role === "user") {
                      return (
                        <div key={idx} className="flex justify-end animate-slide-up">
                          <div className="max-w-[70%] bg-[#5e6ad2] text-white text-base font-medium rounded-2xl rounded-tr-sm px-4 py-3 leading-[1.5]">
                            {msg.text}
                          </div>
                        </div>
                      );
                    }
                    return (
                      <AssistantMessage
                        key={idx}
                        msg={msg}
                        idx={idx}
                        isLatest={isLatest}
                        feedbackGiven={feedbackGiven}
                        onFeedback={handleFeedback}
                        onClarifySelect={handleClarificationSelect}
                        showTextInput={showTextInput}
                        setShowTextInput={setShowTextInput}
                        clarificationAnswer={clarificationAnswer}
                        setClarificationAnswer={setClarificationAnswer}
                        loading={loading}
                      />
                    );
                  })}


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
                        <span className="text-sm text-zinc-400">Running pipeline...</span>
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
                      className="flex-1 bg-[#131316] border border-[#202024] focus:border-[#5e6ad2] focus:outline-none rounded-xl px-4 py-3 text-base text-white placeholder-zinc-500 transition-all"
                    />
                    <button
                      type="submit"
                      disabled={loading || !queryInput.trim()}
                      className="bg-[#5e6ad2] hover:bg-[#707df0] disabled:bg-[#5e6ad2]/30 disabled:cursor-not-allowed text-white font-medium rounded-xl w-12 h-12 flex items-center justify-center transition-all shadow-lg shadow-[#5e6ad2]/20 shrink-0"
                    >
                      {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    </button>
                  </form>
                </div>
              </div>

              {/* RIGHT: Pipeline Execution Trace */}
              <div className={`transition-all duration-300 border-t xl:border-t-0 xl:border-l border-[#202024] bg-[#0c0c0e] ${
                traceCollapsed 
                  ? "w-full xl:w-[60px] h-[50px] xl:h-full shrink-0" 
                  : "w-full xl:w-[360px] h-[300px] xl:h-full shrink-0"
              } flex flex-col overflow-hidden`}>
                {/* Header */}
                <div className={`flex ${traceCollapsed ? "xl:flex-col items-center justify-center p-2 xl:py-6" : "items-center justify-between p-6 pb-4"} border-b border-[#202024]`}>
                  {!traceCollapsed ? (
                    <>
                      <div>
                        <h3 className="text-xs font-semibold text-white mb-1 flex items-center gap-1.5">
                          <Activity className="w-3.5 h-3.5 text-[#5e6ad2]" />
                          LangGraph Pipeline Trace
                        </h3>
                        <p className="text-[10px] text-zinc-400">Observe real-time state flow through nodes</p>
                      </div>
                      <button 
                        type="button"
                        onClick={() => setTraceCollapsed(true)} 
                        title="Collapse panel"
                        className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-all"
                      >
                        <ChevronRight className="w-4 h-4 hidden xl:block" />
                        <ChevronDown className="w-4 h-4 xl:hidden" />
                      </button>
                    </>
                  ) : (
                    <button 
                      type="button"
                      onClick={() => setTraceCollapsed(false)} 
                      title="Expand trace panel"
                      className="p-2 rounded-lg text-[#5e6ad2] hover:text-[#707df0] hover:bg-[#5e6ad2]/10 transition-all flex items-center gap-2 xl:flex-col"
                    >
                      <ChevronLeft className="w-4 h-4 hidden xl:block" />
                      <ChevronUp className="w-4 h-4 xl:hidden" />
                      <Activity className="w-4 h-4" />
                      <span className="xl:hidden text-[10px] font-bold text-zinc-400">Expand Trace</span>
                    </button>
                  )}
                </div>

                {/* Content */}
                {!traceCollapsed && (
                  <div className="flex-1 overflow-y-auto p-6 pr-3 flex flex-col gap-3">
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
                            <p className="flex items-center gap-1.5"><Compass className="w-3.5 h-3.5 text-[#5e6ad2] shrink-0" /> <span>Routed to: <span className="text-white font-bold">{event.update.route}</span></span></p>
                          )}
                          {event.node === "query_decompose" && event.update.sub_questions && (
                            <div className="flex flex-col gap-1">
                              <p className="flex items-center gap-1.5"><Search className="w-3.5 h-3.5 text-zinc-400 shrink-0" /> <span>Decomposed into sub-queries:</span></p>
                              {event.update.sub_questions.map((q: string, i: number) => (
                                <p key={i} className="pl-5 text-zinc-300">- {q}</p>
                              ))}
                            </div>
                          )}
                          {event.node === "retrieve" && event.update.documents && (
                            <p className="flex items-center gap-1.5"><BookOpen className="w-3.5 h-3.5 text-[#5e6ad2] shrink-0" /> <span>Retrieved <span className="text-white font-bold">{event.update.documents.length}</span> child chunks from ChromaDB/BM25.</span></p>
                          )}
                          {event.node === "grade_documents" && event.update.crag_state && (
                            <p className="flex items-center gap-1.5"><BarChart3 className="w-3.5 h-3.5 text-[#10b981] shrink-0" /> <span>CRAG aggregated state: <span className="text-white font-bold">{event.update.crag_state}</span></span></p>
                          )}
                          {event.node === "detect_contradiction" && (
                            <p className="flex items-center gap-1.5"><Scale className="w-3.5 h-3.5 text-zinc-400 shrink-0" /> <span>Contradiction: <span className={event.update.contradiction_found ? "text-[#ef4444]" : "text-[#10b981]"}>{event.update.contradiction_found ? "FOUND" : "Clean"}</span></span></p>
                          )}
                          {event.node === "clarify" && event.update.clarification_question && (
                            <p className="flex items-center gap-1.5"><AlertTriangle className="w-3.5 h-3.5 text-[#f59e0b] shrink-0" /> <span>Asking: <span className="text-[#f59e0b]">{event.update.clarification_question}</span></span></p>
                          )}
                          {event.node === "query_rewrite" && event.update.question && (
                            <p className="flex items-center gap-1.5"><RefreshCw className="w-3.5 h-3.5 text-[#5e6ad2] shrink-0" /> <span>Optimized to: <span className="text-white">{event.update.question}</span></span></p>
                          )}
                          {event.node === "web_search" && (
                            <p className="flex items-center gap-1.5"><Globe className="w-3.5 h-3.5 text-[#10b981] shrink-0" /> <span>Web search executed.</span></p>
                          )}
                          {event.node === "rerank" && event.update.documents && (
                            <p className="flex items-center gap-1.5"><Zap className="w-3.5 h-3.5 text-[#f59e0b] shrink-0" /> <span>Top rerank score: <span className="text-white">{(event.update.documents[0]?.metadata?.rerank_score || 0).toFixed(3)}</span></span></p>
                          )}
                          {event.node === "few_shot_inject" && (
                            <p className="flex items-center gap-1.5"><Package className="w-3.5 h-3.5 text-[#5e6ad2] shrink-0" /> <span>Few-shot injected.</span></p>
                          )}
                          {event.node === "generate" && event.update.generation && (
                            <p className="flex items-center gap-1.5"><FileText className="w-3.5 h-3.5 text-zinc-400 shrink-0" /> <span>"{event.update.generation.slice(0, 60)}..."</span></p>
                          )}
                          {event.node === "grade_hallucination" && (
                            <p className="flex items-center gap-1.5">
                              {event.update.hallucination_free ? <ShieldCheck className="w-3.5 h-3.5 text-[#10b981] shrink-0" /> : <ShieldAlert className="w-3.5 h-3.5 text-[#ef4444] shrink-0" />}
                              <span className={event.update.hallucination_free ? "text-[#10b981]" : "text-[#ef4444]"}>
                                {event.update.hallucination_free ? "Grounded" : "Hallucination!"}
                              </span>
                              <span className="text-zinc-500 font-normal">(Score: {event.update.hallucination_score})</span>
                            </p>
                          )}
                          {event.node === "regenerate" && (
                            <p className="flex items-center gap-1.5"><RefreshCw className="w-3.5 h-3.5 animate-spin text-[#5e6ad2] shrink-0" /> <span>Re-generating (#{event.update.regen_count})</span></p>
                          )}
                          {event.node === "confidence_scorer" && (
                            <p className="flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 text-[#5e6ad2] shrink-0" /> <span>Score: <span className="text-white">{event.update.confidence_score}</span> | Low: {event.update.low_confidence ? "YES" : "NO"}</span></p>
                          )}
                          {event.node === "grade_answer" && (
                            <p className="flex items-center gap-1.5"><Flag className="w-3.5 h-3.5 text-[#10b981] shrink-0" /> <span>Resolves query: <span className="text-white">{event.update.answer_addresses_question ? "YES" : "NO"}</span></span></p>
                          )}
                          {event.node === "pipeline_error" && (
                            <p className="flex items-center gap-1.5"><XCircle className="w-3.5 h-3.5 text-[#ef4444] shrink-0" /> <span className="text-[#ef4444]">{event.update.error}</span></p>
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

                    {/* LLM Telemetry Summary — shown after pipeline completes */}
                    {!streaming && nodeTelemetry.length > 0 && (
                      <div className="border border-[#5e6ad2]/20 bg-[#5e6ad2]/5 rounded-lg p-3 text-[10px] mt-1">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-[#5e6ad2] font-bold uppercase tracking-widest">LLM Call Breakdown</span>
                          <span className="text-zinc-500 font-mono">{totalLlmCalls} calls</span>
                        </div>
                        <div className="flex flex-col gap-1">
                          {nodeTelemetry.map((t: any, i: number) => (
                            <div key={i} className="flex justify-between items-center text-zinc-400 font-mono">
                              <span className="truncate pr-2 text-zinc-300" style={{maxWidth: '55%'}}>{t.node}</span>
                              <div className="flex items-center gap-2">
                                {t.cached && <span className="text-[#10b981] text-[9px] font-bold">CACHE</span>}
                                {t.timeout && <span className="text-[#ef4444] text-[9px] font-bold">TIMEOUT</span>}
                                {t.fallback && <span className="text-[#f59e0b] text-[9px] font-bold">FALLBACK</span>}
                                <span className="text-zinc-200">{t.duration_ms}ms</span>
                              </div>
                            </div>
                          ))}
                          {/* Slowest node callout */}
                          {nodeTelemetry.length > 0 && (() => {
                            const slowest = nodeTelemetry.reduce((a, b) => a.duration_ms > b.duration_ms ? a : b);
                            return (
                              <div className="mt-2 pt-2 border-t border-[#202024] flex justify-between text-zinc-500">
                                <span>Slowest node:</span>
                                <span className="text-[#f59e0b] font-semibold">{slowest.node} ({(slowest.duration_ms/1000).toFixed(2)}s)</span>
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    )}

                    <div ref={traceEndRef} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tab 2: Document Ingestion */}
          {activeTab === "documents" && (
            <div className="flex-1 flex flex-col md:flex-row gap-6 animate-slide-up overflow-y-auto p-1">
              <div className="flex-1 linear-card p-6 flex flex-col gap-4">
                <h2 className="font-instrument text-xl font-bold text-white flex items-center gap-2">
                  <UploadCloud className="w-4 h-4 text-[#5e6ad2]" />
                  Upload Documents
                </h2>
                <p className="text-xs text-zinc-400">
                  Upload text files, markdown, PDFs, Word documents, or image scans. Digital PDFs will be parsed textually; scanned image PDFs automatically route to Tesseract OCR.
                </p>

                <form onSubmit={handleFileUpload} className="flex flex-col gap-4">
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center gap-3 transition-all cursor-pointer select-none ${
                      isDragging
                        ? "border-[#5e6ad2] bg-[#5e6ad2]/10 scale-[1.01]"
                        : "border-[#202024] hover:border-[#5e6ad2] hover:bg-white/[0.01]"
                    }`}
                  >
                    <UploadCloud className={`w-10 h-10 transition-colors ${isDragging ? "text-[#5e6ad2]" : "text-zinc-500"}`} />
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      onChange={(e) => setFiles(e.target.files)}
                      className="hidden"
                    />
                    <p className="text-xs font-medium text-zinc-300">
                      {isDragging ? "Drop your files here!" : "Drag & drop files here, or click to browse"}
                    </p>
                    <p className="text-[10px] text-zinc-500">Supports PDF, DOCX, TXT, PNG, JPG (Max 20MB)</p>
                  </div>

                  {/* Selected files feedback list */}
                  {files && files.length > 0 && (
                    <div className="bg-[#121214] border border-[#202024] rounded-lg p-3 flex flex-col gap-2">
                      <div className="flex justify-between items-center border-b border-[#202024] pb-1.5 mb-1">
                        <span className="text-[10px] uppercase font-bold text-zinc-400">Selected Files ({files.length})</span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setFiles(null);
                            if (fileInputRef.current) fileInputRef.current.value = "";
                          }}
                          className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors"
                        >
                          Clear Selection
                        </button>
                      </div>
                      <div className="max-h-32 overflow-y-auto flex flex-col gap-1.5 pr-1">
                        {Array.from(files).map((f, fi) => (
                          <div key={fi} className="flex justify-between items-center text-xs text-zinc-300 bg-[#09090b] px-2.5 py-1.5 rounded border border-[#1a1a1e]">
                            <span className="truncate pr-4">{f.name}</span>
                            <span className="text-[10px] text-zinc-500 shrink-0">{(f.size / (1024 * 1024)).toFixed(2)} MB</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-start gap-2.5 px-1 py-1.5 border border-[#202024]/50 bg-[#121214]/50 rounded-lg p-2.5">
                    <input
                      type="checkbox"
                      id="keepPersistent"
                      checked={keepPersistent}
                      onChange={(e) => setKeepPersistent(e.target.checked)}
                      className="mt-0.5 w-3.5 h-3.5 rounded border-[#202024] text-[#5e6ad2] bg-[#09090b] focus:ring-[#5e6ad2] focus:ring-offset-0 focus:ring-1 cursor-pointer"
                    />
                    <div className="flex flex-col gap-0.5">
                      <label htmlFor="keepPersistent" className="text-xs font-semibold text-zinc-200 cursor-pointer select-none">
                        Persistent Document (Keep forever)
                      </label>
                      <span className="text-[10px] text-zinc-500">
                        If unchecked, the file will be scoped to this session only and will not affect other conversations.
                      </span>
                    </div>
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
                  <p className="text-xs text-zinc-300 font-medium bg-[#121214] border border-[#202024] p-3 rounded-lg flex items-center gap-1.5">
                    <Info className="w-4 h-4 text-[#5e6ad2] shrink-0" />
                    <span>{uploadStatus}</span>
                  </p>
                )}

                {uploadResults.length > 0 && (
                  <div>
                    <h3 className="font-instrument text-xl font-bold text-zinc-300 mb-2">Ingestion Results</h3>
                    <div className="flex flex-col gap-2">
                      {uploadResults.map((r, idx) => (
                        <div key={idx} className="bg-[#0c0c0e] border border-[#202024] rounded-lg p-3 text-xs">
                          <p className="font-semibold text-zinc-200">{r.file}</p>
                          {r.status === "ingested" ? (
                            <p className="text-[#10b981] mt-1 flex items-center gap-1.5">
                              <Check className="w-3.5 h-3.5 shrink-0" />
                              <span>Successfully indexed {r.summary.children} chunks ({r.summary.parents} parent contexts) in {r.summary.seconds}s.</span>
                            </p>
                          ) : (
                            <p className="text-[#ef4444] mt-1 flex items-center gap-1.5">
                              <XCircle className="w-3.5 h-3.5 shrink-0" />
                              <span>Ingestion failed: {r.error}</span>
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>


            </div>
          )}

          {/* Tab 3: System Analytics & Eval */}
          {activeTab === "analytics" && (
            <div className="flex-1 flex flex-col gap-6 animate-slide-up overflow-y-auto p-1">
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
                <div className="flex justify-between items-center">
                  <h3 className="font-instrument text-xl font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-[#5e6ad2]" />
                    Learned Few-Shot Memory Examples
                  </h3>
                  {stats.few_shot_examples > 0 && (
                    <button
                      onClick={handleClearFewShot}
                      className="border border-[#ef4444]/30 hover:border-[#ef4444] text-[#ef4444] hover:bg-[#ef4444]/5 font-medium rounded-lg px-3 py-1.5 text-[10px] transition-all"
                    >
                      Reset Memory
                    </button>
                  )}
                </div>
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
                <h3 className="font-instrument text-xl font-bold text-white">Evaluation Benchmarking</h3>
                <p className="text-xs text-zinc-400">
                  Execute the built-in 12-question benchmark suite comparing baseline standard RAG against the adaptive self-correcting RAG. (Warning: this executes 24 full pipeline pipelines and takes up to 2 minutes).
                </p>
                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={runEvaluation}
                    disabled={evalLoading}
                    className="bg-[#5e6ad2] hover:bg-[#707df0] disabled:bg-[#5e6ad2]/40 disabled:cursor-not-allowed text-white font-medium rounded-lg px-5 py-2.5 text-xs transition-all flex items-center gap-2"
                  >
                    {evalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
                    Run Evaluation Suite
                  </button>
                  <button
                    onClick={async () => {
                      setEvalLoading(true);
                      setEvalResults(null);
                      try {
                        const res = await axios.get("/api/evaluate/extended");
                        setEvalResults({ ...res.data, extended: true });
                      } catch (err) {
                        console.error("Extended eval failed", err);
                      } finally {
                        setEvalLoading(false);
                      }
                    }}
                    disabled={evalLoading}
                    className="border border-[#5e6ad2]/30 hover:border-[#5e6ad2] text-[#5e6ad2] hover:bg-[#5e6ad2]/10 disabled:opacity-40 font-medium rounded-lg px-5 py-2.5 text-xs transition-all flex items-center gap-2"
                  >
                    {evalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
                    Extended Eval (P@K · NDCG · Faithfulness)
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
