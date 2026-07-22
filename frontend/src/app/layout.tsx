import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Ultimate Self-Correcting RAG",
  description: "Advanced agentic RAG pipeline with automatic contradiction detection, ambiguity clarification, and hallucination correction.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${plusJakartaSans.variable} h-full antialiased`}>
      <body className="min-h-full bg-[#09090b] text-[#fafafa] flex flex-col">
        {children}
      </body>
    </html>
  );
}
