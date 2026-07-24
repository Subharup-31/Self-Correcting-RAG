"use client";

import React, { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";

export default function ArchitectureSection() {
  const [scale, setScale] = useState(1);
  const [height, setHeight] = useState(680);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  useEffect(() => {
    const handleResize = () => {
      if (canvasWrapperRef.current) {
        let width = canvasWrapperRef.current.offsetWidth;
        if (width <= 0) {
          // Fallback to window innerWidth or default 1152px max-w
          width = typeof window !== "undefined" ? Math.min(1152, window.innerWidth - 48) : 1152;
        }
        if (width > 0) {
          const scaleVal = Math.min(1, width / 1300);
          setScale(scaleVal);
          setHeight(680 * scaleVal);
        }
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);

    // Post-hydration layout check to prevent cutoff
    const timer = setTimeout(handleResize, 200);

    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, []);

  const nodes = [
    {
      id: "user",
      badge: "Q",
      badgeColor: "text-blue-400 bg-blue-500/10 border-blue-500/20",
      title: "User Query",
      desc: "NL Input Request",
      left: "3%",
      top: "43%",
      width: "165px"
    },
    {
      id: "router",
      badge: "R",
      badgeColor: "text-indigo-400 bg-indigo-500/10 border-indigo-500/20",
      title: "Adaptive Router",
      desc: "Intent Classification",
      left: "19%",
      top: "43%",
      width: "165px"
    },
    {
      id: "hyde",
      badge: "H",
      badgeColor: "text-violet-400 bg-violet-500/10 border-violet-500/20",
      title: "HyDE Reformulation",
      desc: "Hypothetical Context",
      left: "35%",
      top: "27%",
      width: "165px"
    },
    {
      id: "retrieval",
      badge: "V",
      badgeColor: "text-teal-400 bg-teal-500/10 border-teal-500/20",
      title: "Hybrid Search",
      desc: "Vector + BM25 + RRF",
      left: "51%",
      top: "27%",
      width: "165px"
    },
    {
      id: "grader",
      badge: "G",
      badgeColor: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
      title: "Document Grader",
      desc: "CRAG Relevance Score",
      left: "67%",
      top: "27%",
      width: "165px"
    },
    {
      id: "rewrite",
      badge: "L",
      badgeColor: "text-amber-400 bg-amber-500/10 border-amber-500/20",
      title: "Query Rewrite",
      desc: "Loop Reformulation",
      left: "51%",
      top: "9%",
      width: "165px"
    },
    {
      id: "web",
      badge: "W",
      badgeColor: "text-orange-400 bg-orange-500/10 border-orange-500/20",
      title: "Web Fallback",
      desc: "Tavily Search API",
      left: "51%",
      top: "47%",
      width: "165px"
    },
    {
      id: "clarify",
      badge: "C",
      badgeColor: "text-yellow-400 bg-yellow-500/10 border-yellow-500/20",
      title: "Clarification Gate",
      desc: "Solicit Human Input",
      left: "67%",
      top: "9%",
      width: "165px"
    },
    {
      id: "contradiction",
      badge: "E",
      badgeColor: "text-pink-400 bg-pink-500/10 border-pink-500/20",
      title: "Contradiction Resolver",
      desc: "Pairwise Fact Check",
      left: "83%",
      top: "27%",
      width: "165px"
    },
    {
      id: "memory",
      badge: "M",
      badgeColor: "text-purple-400 bg-purple-500/10 border-purple-500/20",
      title: "Few-Shot Memory",
      desc: "Adapts to Feedback",
      left: "35%",
      top: "72%",
      width: "165px"
    },
    {
      id: "generator",
      badge: "A",
      badgeColor: "text-sky-400 bg-sky-500/10 border-sky-500/20",
      title: "Response Generator",
      desc: "Context Injection",
      left: "51%",
      top: "72%",
      width: "165px"
    },
    {
      id: "grounding",
      badge: "F",
      badgeColor: "text-red-400 bg-red-500/10 border-red-500/20",
      title: "Grounding Grader",
      desc: "Hallucination Gate",
      left: "67%",
      top: "72%",
      width: "165px"
    },
    {
      id: "outcome",
      badge: "O",
      badgeColor: "text-green-400 bg-green-500/10 border-green-500/20",
      title: "Verified Output",
      desc: "Grounded Answer + Trace",
      left: "83%",
      top: "72%",
      width: "165px"
    }
  ];

  return (
    <section 
      ref={containerRef}
      className="relative bg-black py-28 md:py-40 px-6 overflow-hidden flex flex-col items-center justify-center border-t border-white/5"
    >
      {/* Background radial highlight */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(94,106,210,0.03)_0%,_transparent_75%)] pointer-events-none" />

      <div className="w-full max-w-6xl relative z-10">
        {/* Title */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <span className="text-white/40 text-xs tracking-widest uppercase mb-4 block">
            Agentic Data Flow
          </span>
          <h2 className="text-4xl md:text-6xl text-white tracking-tight font-normal">
            System{" "}
            <span 
              className="text-white/40 italic font-normal"
              style={{ fontFamily: "'Instrument Serif', serif" }}
            >
              Architecture
            </span>
          </h2>
        </motion.div>

        {/* Mac OS Styled Canvas Window */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
          transition={{ duration: 0.8, delay: 0.15 }}
          className="w-full bg-[#09090b] border border-[#202024] rounded-2xl overflow-hidden shadow-2xl relative"
        >
          {/* Mac Window Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#202024] bg-[#0c0c0e]">
            {/* Window control dots */}
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-[#ef4444]" />
              <div className="w-3 h-3 rounded-full bg-[#f59e0b]" />
              <div className="w-3 h-3 rounded-full bg-[#10b981]" />
            </div>
            {/* File Path Title */}
            <span className="text-zinc-500 font-mono text-xs select-none">
              ultimate-rag / pipeline-flow
            </span>
            {/* Empty space for alignment */}
            <div className="w-14" />
          </div>

          {/* Canvas Body (No Scroll, Scaled to Fit) */}
          <div 
            ref={canvasWrapperRef} 
            className="w-full select-none overflow-hidden relative"
            style={{ height: `${height}px` }}
          >
            {/* Grid Area with Fixed Dimensions, Scaled Dynamically */}
            <div 
              className="relative w-[1300px] h-[680px] bg-[#09090b] origin-top-left"
              style={{
                transform: `scale(${scale})`,
                backgroundImage: `
                  linear-gradient(to right, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                  linear-gradient(to bottom, rgba(255, 255, 255, 0.015) 1px, transparent 1px)
                `,
                backgroundSize: "24px 24px"
              }}
            >
              {/* SVG Connective Lines Overlay */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                <defs>
                  {/* Arrow marker definitions */}
                  <marker
                    id="arrow-blue"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
                  </marker>
                  <marker
                    id="arrow-gray"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#3f3f46" />
                  </marker>
                  <marker
                    id="arrow-emerald"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
                  </marker>
                  <marker
                    id="arrow-amber"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
                  </marker>
                  <marker
                    id="arrow-red"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444" />
                  </marker>
                </defs>

                {/* Connections */}
                {/* User -> Router (Gray connection) */}
                <path 
                  d="M 204 323.4 L 247 323.4" 
                  fill="none" 
                  stroke="#3f3f46" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-gray)" 
                />

                {/* Router -> HyDE (RAG Path - Indigo) */}
                <path 
                  d="M 412 323.4 L 435 323.4 L 435 214.6 L 455 214.6" 
                  fill="none" 
                  stroke="#6366f1" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* Router -> Web Search Fallback (Direct Search - Orange) */}
                <path 
                  d="M 412 323.4 L 435 323.4 L 435 350.6 L 663 350.6" 
                  fill="none" 
                  stroke="#f97316" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* HyDE -> Hybrid Search (Vector + BM25) */}
                <path 
                  d="M 620 214.6 L 663 214.6" 
                  fill="none" 
                  stroke="#6366f1" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* Hybrid Search -> Document Grader */}
                <path 
                  d="M 828 214.6 L 871 214.6" 
                  fill="none" 
                  stroke="#6366f1" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* Document Grader -> Contradiction Resolver (Correct Path - Emerald) */}
                <path 
                  d="M 1036 214.6 L 1079 214.6" 
                  fill="none" 
                  stroke="#10b981" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-emerald)" 
                />

                {/* Document Grader -> Query Rewrite Loop (Incorrect Path - Amber) */}
                <path 
                  d="M 953.5 183.6 L 953.5 92.2 L 828 92.2" 
                  fill="none" 
                  stroke="#f59e0b" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-amber)" 
                />

                {/* Query Rewrite Loop -> Hybrid Search (Loop reconnects) */}
                <path 
                  d="M 663 92.2 L 590 92.2 L 590 180 L 590 214.6 L 663 214.6" 
                  fill="none" 
                  stroke="#f59e0b" 
                  strokeWidth="2" 
                  strokeDasharray="4"
                  markerEnd="url(#arrow-amber)" 
                />

                {/* Document Grader -> Clarification Gate (Ambiguous Path - Yellow) */}
                <path 
                  d="M 953.5 183.6 L 953.5 123" 
                  fill="none" 
                  stroke="#eab308" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-amber)" 
                />

                {/* Clarification Gate -> Solicit loop */}
                <path 
                  d="M 1036 92.2 L 1055 92.2 L 1055 50" 
                  fill="none" 
                  stroke="#eab308" 
                  strokeWidth="1.5" 
                  strokeDasharray="4"
                />

                {/* Contradiction Resolver -> Generator (Verified Contexts) */}
                <path 
                  d="M 1161.5 245.6 L 1161.5 450 L 745.5 450 L 745.5 489.6" 
                  fill="none" 
                  stroke="#10b981" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-emerald)" 
                />

                {/* Web Search Fallback -> Generator */}
                <path 
                  d="M 828 350.6 L 850 350.6 L 850 450 L 745.5 450 L 745.5 489.6" 
                  fill="none" 
                  stroke="#f97316" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* Few-Shot Memory -> Generator */}
                <path 
                  d="M 620 520.6 L 663 520.6" 
                  fill="none" 
                  stroke="#8b5cf6" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-blue)" 
                />

                {/* Generator -> Grounding Grader (Fact ground audit) */}
                <path 
                  d="M 828 520.6 L 871 520.6" 
                  fill="none" 
                  stroke="#ef4444" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-red)" 
                />

                {/* Grounding Grader -> Verified Output (Grounded Pass - Green) */}
                <path 
                  d="M 1036 520.6 L 1079 520.6" 
                  fill="none" 
                  stroke="#22c55e" 
                  strokeWidth="2" 
                  markerEnd="url(#arrow-emerald)" 
                />

                {/* Grounding Grader -> Generator (Hallucinated Loop - Red) */}
                <path 
                  d="M 953.5 551.6 L 953.5 585 L 745.5 585 L 745.5 551.6" 
                  fill="none" 
                  stroke="#ef4444" 
                  strokeWidth="2" 
                  strokeDasharray="4"
                  markerEnd="url(#arrow-red)" 
                />
              </svg>

              {/* Floating Diagram Nodes */}
              {nodes.map((node) => (
                <div
                  key={node.id}
                  className="absolute bg-[#121214] border border-[#202024] rounded-2xl p-3 flex items-center gap-3 shadow-xl transition-all duration-300 hover:border-zinc-500 z-10"
                  style={{
                    left: node.left,
                    top: node.top,
                    width: node.width,
                    height: "62px"
                  }}
                >
                  {/* Badge */}
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs border shrink-0 ${node.badgeColor}`}>
                    {node.badge}
                  </div>
                  {/* Title & Desc */}
                  <div className="min-w-0">
                    <h4 className="text-zinc-200 text-[11px] font-semibold tracking-tight truncate leading-tight">
                      {node.title}
                    </h4>
                    <span className="text-[9px] text-zinc-500 tracking-tight block truncate mt-0.5 leading-tight">
                      {node.desc}
                    </span>
                  </div>
                </div>
              ))}

              {/* Bottom command bar pill (Matches attached image styling) */}
              <div className="absolute bottom-6 left-6 bg-[#121214] border border-[#202024] rounded-full px-5 py-2 flex items-center gap-2.5 shadow-lg max-w-sm">
                <div className="w-5 h-5 rounded bg-blue-500 flex items-center justify-center shrink-0">
                  <span className="text-white text-[10px] font-bold">↳</span>
                </div>
                <span className="text-zinc-400 font-mono text-[11px] tracking-tight">
                  self-healing agents resolve hallucinations in real time
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
