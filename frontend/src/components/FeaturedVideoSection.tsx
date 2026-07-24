"use client";

import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";

import Link from "next/link";

export default function FeaturedVideoSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  return (
    <section 
      ref={containerRef}
      className="bg-black pt-6 md:pt-10 pb-20 md:pb-32 px-6 overflow-hidden flex justify-center"
    >
      <motion.div 
        initial={{ opacity: 0, y: 60 }}
        animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 60 }}
        transition={{ duration: 0.9 }}
        className="relative w-full max-w-6xl rounded-3xl overflow-hidden aspect-video group"
      >
        {/* Video */}
        <video 
          className="w-full h-full object-cover"
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260402_054547_9875cfc5-155a-4229-8ec8-b7ba7125cbf8.mp4"
          muted
          autoPlay
          loop
          playsInline
          preload="auto"
        />

        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none" />

        {/* Bottom Overlay Content */}
        <div className="absolute bottom-0 left-0 right-0 p-6 md:p-10 flex flex-col md:flex-row md:items-end justify-between gap-6 z-10">
          {/* Left card */}
          <div className="liquid-glass rounded-2xl p-6 md:p-8 max-w-md">
            <span className="block text-white/50 text-xs tracking-widest uppercase mb-3">
              Contradiction Resolution
            </span>
            <p className="text-white text-sm md:text-base leading-relaxed">
              We actively eliminate contradictions. Our pipeline checks retrieved documents pairwise, grading facts, flagging discrepancies, and resolving ambiguities before generating responses.
            </p>
          </div>

          {/* Right button */}
          <div>
            <motion.div 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-block"
            >
              <Link 
                href="/dashboard" 
                className="liquid-glass rounded-full px-8 py-3 text-white text-sm font-medium transition-colors hover:bg-white/5 cursor-pointer block text-center"
              >
                Launch Sandbox
              </Link>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
