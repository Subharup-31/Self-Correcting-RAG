"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { Globe, ArrowRight } from "lucide-react";
import AboutSection from "../components/AboutSection";
import FeaturedVideoSection from "../components/FeaturedVideoSection";
import PhilosophySection from "../components/PhilosophySection";
import ServicesSection from "../components/ServicesSection";
import ArchitectureSection from "../components/ArchitectureSection";

const GithubIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    {...props}
  >
    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
  </svg>
);

export default function LandingPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [queryIndex, setQueryIndex] = useState(0);
  const [displayText, setDisplayText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let animFrameId: number;
    let fadeStart: number | null = null;
    let fadeMode: "in" | "out" | null = null;

    const animateFade = (timestamp: number) => {
      if (!fadeStart) fadeStart = timestamp;
      const progress = timestamp - fadeStart;
      const duration = 500; // 500ms fade duration

      if (fadeMode === "in") {
        const opacity = Math.min(progress / duration, 1);
        video.style.opacity = opacity.toString();
        if (opacity < 1) {
          animFrameId = requestAnimationFrame(animateFade);
        } else {
          fadeMode = null;
          fadeStart = null;
        }
      } else if (fadeMode === "out") {
        const opacity = Math.max(1 - progress / duration, 0);
        video.style.opacity = opacity.toString();
        if (opacity > 0) {
          animFrameId = requestAnimationFrame(animateFade);
        } else {
          fadeMode = null;
          fadeStart = null;
        }
      }
    };

    const handleCanPlay = () => {
      video.play().catch(err => console.log("Autoplay blocked:", err));
      fadeMode = "in";
      fadeStart = null;
      animFrameId = requestAnimationFrame(animateFade);
    };

    const handleTimeUpdate = () => {
      const remaining = video.duration - video.currentTime;
      // Triggers fade out 0.55s before video ends
      if (remaining <= 0.55 && fadeMode !== "out" && video.style.opacity !== "0") {
        fadeMode = "out";
        fadeStart = null;
        animFrameId = requestAnimationFrame(animateFade);
      }
    };

    const handleEnded = () => {
      video.style.opacity = "0";
      // Wait 100ms, reset currentTime, play again, fade back in
      setTimeout(() => {
        video.currentTime = 0;
        video.play().then(() => {
          fadeMode = "in";
          fadeStart = null;
          animFrameId = requestAnimationFrame(animateFade);
        }).catch(err => console.log("Loop play failed:", err));
      }, 100);
    };

    // If video is already ready to play
    if (video.readyState >= 3) {
      handleCanPlay();
    }

    video.addEventListener("canplay", handleCanPlay);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("ended", handleEnded);

    return () => {
      video.removeEventListener("canplay", handleCanPlay);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("ended", handleEnded);
      cancelAnimationFrame(animFrameId);
    };
  }, []);

  const COMPLEX_QUERIES = [
    "What is HyDE and how does it improve retrieval precision?",
    "Are there contradictory terms in the Q4 financial agreements?",
    "How does the CRAG node grade retrieved context quality?",
    "Summarize the technical manual's section on system failure recovery.",
    "Check the ingested documents for Q3 performance metrics conflicts."
  ];

  useEffect(() => {
    const fullText = COMPLEX_QUERIES[queryIndex];
    let timer: NodeJS.Timeout;

    if (isDeleting) {
      timer = setTimeout(() => {
        setDisplayText((prev) => prev.slice(0, -1));
      }, 25);
    } else {
      timer = setTimeout(() => {
        setDisplayText(fullText.slice(0, displayText.length + 1));
      }, 40);
    }

    if (!isDeleting && displayText === fullText) {
      timer = setTimeout(() => setIsDeleting(true), 3500);
    }

    if (isDeleting && displayText === "") {
      setIsDeleting(false);
      setQueryIndex((prev) => (prev + 1) % COMPLEX_QUERIES.length);
    }

    return () => clearTimeout(timer);
  }, [displayText, isDeleting, queryIndex]);

  return (
    <div className="bg-black text-white min-h-screen flex flex-col font-sans select-none overflow-x-hidden">
      {/* SECTION 1: HERO CONTAINER */}
      <header className="min-h-screen relative overflow-hidden flex flex-col justify-between w-full">
        {/* Background video */}
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover object-bottom pointer-events-none transition-none"
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260405_074625_a81f018a-956b-43fb-9aee-4d1508e30e6a.mp4"
          muted
          autoPlay
          playsInline
          preload="auto"
          style={{ opacity: 0 }}
        />

        {/* Backdrop overlay */}
        <div className="absolute inset-0 bg-black/40 pointer-events-none" />

        {/* Navbar */}
        <nav className="relative z-20 px-6 py-6 w-full">
          <div className="liquid-glass rounded-full max-w-5xl mx-auto px-6 py-3 flex items-center justify-between">
            {/* Left Brand */}
            <div className="flex items-center gap-2">
              <Globe className="w-6 h-6 text-white" />
              <span className="font-semibold text-lg text-white tracking-tight">Ultimate RAG</span>
            </div>

            {/* Center Navigation Links (Hidden on mobile) */}
            <div className="hidden md:flex items-center gap-8 ml-8">
              <a href="#features" className="text-white/80 hover:text-white text-sm font-medium transition-colors">
                Pipeline Features
              </a>
              <a href="#about" className="text-white/80 hover:text-white text-sm font-medium transition-colors">
                Architecture
              </a>
              <Link href="/dashboard" className="text-white/80 hover:text-white text-sm font-medium transition-colors flex items-center gap-1.5">
                <span>Launch App</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {/* Right CTAs */}
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="md:hidden text-white/80 hover:text-white text-xs font-semibold mr-2 transition-colors">
                App Sandbox
              </Link>
              <Link href="/dashboard" className="liquid-glass rounded-full px-6 py-2 text-white text-sm font-medium hover:bg-white/5 transition-colors cursor-pointer">
                Launch
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero Content */}
        <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-6 py-12 text-center -translate-y-[10%] md:-translate-y-[15%] max-w-4xl mx-auto w-full">
          {/* Main Heading */}
          <h1 
            className="text-5xl md:text-7xl lg:text-8xl text-white tracking-tight font-normal leading-[1.05] mb-8"
            style={{ fontFamily: "'Instrument Serif', serif" }}
          >
            Retrieve without <span className="italic">hallucination</span>.
          </h1>

          {/* Simulated Query Typing Box */}
          <div className="max-w-xl w-full mb-6">
            <div className="liquid-glass rounded-full pl-6 pr-2 py-2 flex items-center justify-between gap-3 border border-white/5 text-left">
              <div className="flex-1 text-white text-sm flex items-center overflow-hidden h-9">
                <span className="truncate text-white/90 font-light select-none">
                  {displayText || <span className="text-white/30">Ask our agentic RAG...</span>}
                </span>
                <span className="inline-block w-[2px] h-4 bg-white/80 ml-1 animate-pulse" />
              </div>
              <div className="bg-white rounded-full p-3 text-black flex items-center justify-center shrink-0">
                <ArrowRight className="w-5 h-5" />
              </div>
            </div>
          </div>

          {/* Subtitle */}
          <p className="text-white/70 text-xs md:text-sm leading-relaxed max-w-lg mb-8">
            A production-ready agentic RAG pipeline that ingests messy documents, resolves context contradictions, and self-corrects hallucinations in real time before answering.
          </p>

          {/* Manifesto Button */}
          <Link href="/dashboard" className="liquid-glass rounded-full px-8 py-3 text-white text-sm font-medium hover:bg-white/5 transition-colors cursor-pointer">
            Explore Dashboard Sandbox
          </Link>
        </div>

        {/* Social Icons Footer */}
        <div className="relative z-10 flex justify-center pb-12">
          <a
            href="https://github.com/Subharup-31/Self-Correcting-RAG"
            target="_blank"
            rel="noopener noreferrer"
            className="liquid-glass rounded-full px-6 py-3 flex items-center gap-2 text-white/80 hover:text-white hover:bg-white/5 transition-all cursor-pointer text-sm font-medium"
          >
            <GithubIcon className="w-5 h-5" />
            <span>GitHub Repository</span>
          </a>
        </div>
      </header>

      {/* SECTION 2: ABOUT */}
      <div id="about">
        <AboutSection />
      </div>

      {/* SECTION 3: FEATURED VIDEO */}
      <div id="features">
        <FeaturedVideoSection />
      </div>

      {/* SECTION 4: PHILOSOPHY */}
      <PhilosophySection />

      {/* SECTION: ARCHITECTURE */}
      <ArchitectureSection />

      {/* SECTION 5: SERVICES */}
      <ServicesSection />
    </div>
  );
}
