"use client";

import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";

export default function PhilosophySection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  return (
    <section 
      ref={containerRef}
      className="bg-black py-28 md:py-40 px-6 overflow-hidden flex justify-center"
    >
      <div className="w-full max-w-6xl">
        {/* Heading */}
        <motion.h2 
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-7xl lg:text-8xl text-white tracking-tight mb-16 md:mb-24"
        >
          Retrieval{" "}
          <span 
            className="text-white/40 italic font-normal"
            style={{ fontFamily: "'Instrument Serif', serif" }}
          >
            then x
          </span>{" "}
          Precision
        </motion.h2>

        {/* Two-column Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 items-center">
          {/* Left Column: Video */}
          <motion.div 
            initial={{ opacity: 0, x: -40 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -40 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="rounded-3xl overflow-hidden aspect-[4/3] bg-zinc-900"
          >
            <video 
              className="w-full h-full object-cover"
              src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260307_083826_e938b29f-a43a-41ec-a153-3d4730578ab8.mp4"
              muted
              autoPlay
              loop
              playsInline
              preload="auto"
            />
          </motion.div>

          {/* Right Column: Text blocks */}
          <motion.div 
            initial={{ opacity: 0, x: 40 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 40 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-col gap-8 md:gap-12"
          >
            {/* Block 1 */}
            <div>
              <span className="block text-white/40 text-xs tracking-widest uppercase mb-4">
                Hybrid Ingestion & HyDE
              </span>
              <p className="text-white/70 text-base md:text-lg leading-relaxed">
                Our ingestion engine fuses lexical BM25 with BGE-M3 vector search. We resolve query-document semantic gaps by generating a hypothetical answer (HyDE) first, then retrieve chunks across digital and messy scanned OCR documents.
              </p>
            </div>

            {/* Divider */}
            <div className="w-full h-px bg-white/10" />

            {/* Block 2 */}
            <div>
              <span className="block text-white/40 text-xs tracking-widest uppercase mb-4">
                Self-Healing Query Loops
              </span>
              <p className="text-white/70 text-base md:text-lg leading-relaxed">
                When initial document grading yields low confidence or ambiguous context, our LangGraph router triggers query rewrites and retrieves again. If context remains insufficient, it automatically falls back to live web searches.
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
