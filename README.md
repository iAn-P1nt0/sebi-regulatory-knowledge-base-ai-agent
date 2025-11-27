# SEBI Regulatory Research Agent

Agentic AI toolkit focused on researching Securities and Exchange Board of India (SEBI) regulations using Retrieval-Augmented Generation (RAG) primitives, LangChain.js, and LangGraph.

## Features
- Modular architecture for configuration, corpus ingestion, and LangChain toolchains
- Qdrant vector support for regulatory corpora
- Ready-to-extend LangGraph agents for compliance workflows

## Getting Started
1. **Install dependencies**
   ```bash
   npm install
   ```
2. **Copy environment variables**
   ```bash
   cp .env.example .env
   ```
   Fill in `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and adjust `NODE_ENV` as needed.
3. **Run in watch mode**
   ```bash
   npm run dev
   ```

## Available Scripts
- `npm run dev` – Local TS watch via tsx
- `npm run build` – Compile TypeScript to `dist/`
- `npm run test` – Execute Vitest suite
- `npm run lint` – Run ESLint across the repo

## Project Structure
```
src/
  config/
  corpus/
  tools/
  agents/
  utils/
  types/
  index.ts

tests/
  corpus/
  tools/
  agents/
```

## Requirements
- Node.js >= 22
- TypeScript 5.7+
- OpenAI & Qdrant credentials for agent execution
