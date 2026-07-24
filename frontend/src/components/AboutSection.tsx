"use client";

import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";

export default function AboutSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  return (
    <section 
      ref={containerRef}
      className="relative bg-black pt-32 md:pt-44 pb-10 md:pb-14 px-6 overflow-hidden"
    >
      {/* Subtle radial gradient overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(255,255,255,0.03)_0%,_transparent_70%)] pointer-events-none" />

      <div className="relative max-w-6xl mx-auto flex flex-col items-start z-10">
        {/* Label */}
        <motion.span 
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.6 }}
          className="text-white/40 text-sm tracking-widest uppercase mb-6"
        >
          Core Paradigm
        </motion.span>

        {/* Heading */}
        <motion.h2 
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="text-4xl md:text-6xl lg:text-7xl text-white leading-[1.1] tracking-tight max-w-5xl"
        >
          <span 
            className="text-white/60 italic" 
            style={{ fontFamily: "'Instrument Serif', serif" }}
          >
            Pioneering self-corrective retrieval
          </span>{" "}
          for{" "}
          <br className="hidden md:block" />
          <span 
            className="text-white/60 italic" 
            style={{ fontFamily: "'Instrument Serif', serif" }}
          >
            production systems that require absolute truth.
          </span>
        </motion.h2>
      </div>
    </section>
  );
}
