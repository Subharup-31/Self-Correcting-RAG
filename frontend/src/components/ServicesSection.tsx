"use client";

import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

export default function ServicesSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  const cards = [
    {
      videoSrc: "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4",
      tag: "Verification",
      title: "Hallucination Grading",
      description: "Every response is audited using LangGraph-powered fact-checkers. If any statement lacks evidentiary support in the retrieved files, the engine rewrites query contexts and regenerates."
    },
    {
      videoSrc: "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260324_151826_c7218672-6e92-402c-9e45-f1e0f454bdc4.mp4",
      tag: "Optimizations",
      title: "Dynamic Few-Shot Exemplars",
      description: "The system logs high-quality execution paths and user feedback. When a similar query is processed, those exemplars are injected as few-shot prompts to boost reasoning accuracy."
    }
  ];

  return (
    <section 
      ref={containerRef}
      className="relative bg-black py-28 md:py-40 px-6 overflow-hidden flex justify-center"
    >
      {/* Subtle radial gradient overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(255,255,255,0.02)_0%,_transparent_60%)] pointer-events-none" />

      <div className="w-full max-w-6xl relative z-10">
        {/* Header Row */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.7 }}
          className="flex justify-between items-end mb-12 md:mb-16 border-b border-white/10 pb-6"
        >
          <h2 className="text-3xl md:text-5xl text-white tracking-tight">
            Pipeline Capabilities
          </h2>
          <span className="text-white/40 text-sm hidden md:inline uppercase tracking-widest">
            Core Modules
          </span>
        </motion.div>

        {/* Two-card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
          {cards.map((card, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
              transition={{ duration: 0.8, delay: idx * 0.15 }}
              className="liquid-glass rounded-3xl overflow-hidden group cursor-pointer"
            >
              {/* Card video area */}
              <div className="aspect-video overflow-hidden relative">
                <video 
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
                  src={card.videoSrc}
                  muted
                  autoPlay
                  loop
                  playsInline
                  preload="auto"
                />
                {/* Gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />
              </div>

              {/* Card body */}
              <div className="p-6 md:p-8 flex flex-col gap-4">
                <div className="flex justify-between items-center">
                  <span className="text-white/40 text-xs uppercase tracking-widest font-medium">
                    {card.tag}
                  </span>
                  
                  {/* Arrow icon */}
                  <div className="liquid-glass rounded-full p-2 text-white/80 group-hover:text-white transition-colors">
                    <ArrowUpRight className="w-4 h-4" />
                  </div>
                </div>

                <div>
                  <h3 className="text-white text-xl md:text-2xl mb-3 tracking-tight font-semibold">
                    {card.title}
                  </h3>
                  <p className="text-white/50 text-sm leading-relaxed">
                    {card.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
