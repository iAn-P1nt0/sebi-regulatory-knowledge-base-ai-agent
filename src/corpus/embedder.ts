import { OpenAIEmbeddings } from '@langchain/openai';
import NodeCache from 'node-cache';
import crypto from 'node:crypto';

import { type SEBIChunk, SEBIChunkSchema } from '../types/sebi-document.js';

const DEFAULT_MODEL = 'text-embedding-3-large';
const VECTOR_SIZE = 3072;
const DEFAULT_BATCH_SIZE = 100;
const DEFAULT_MAX_CONCURRENCY = 5;
const MAX_CACHE_ENTRIES = 1000;
const DEFAULT_RATE_LIMIT_PER_MINUTE = 3000;

const ONE_MINUTE_MS = 60_000;

export interface EmbeddingGeneratorOptions {
  model?: string;
  batchSize?: number;
  maxConcurrency?: number;
  cacheTtlSeconds?: number;
  rateLimitPerMinute?: number;
  logger?: Pick<typeof console, 'info' | 'warn' | 'error'>;
}

export interface EmbeddingResult {
  embeddings: number[][];
}

/**
 * Generate an MD5 based cache key for the provided text.
 */
export function cacheKey(text: string): string {
  return crypto.createHash('md5').update(text).digest('hex');
}

/**
 * Utility helper to pause execution.
 */
async function delay(ms: number): Promise<void> {
  if (ms <= 0) return;
  await new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Embedding generator with caching, batching, and rate limiting support.
 */
export class EmbeddingGenerator {
  private readonly embeddings: OpenAIEmbeddings;
  private readonly cache: NodeCache;
  private readonly cacheSize: number;
  private readonly batchSize: number;
  private readonly maxConcurrency: number;
  private readonly logger: Pick<typeof console, 'info' | 'warn' | 'error'>;
  private readonly rateLimitPerMinute: number;
  private readonly requestTimestamps: number[] = [];

  constructor(private readonly apiKey: string, options: EmbeddingGeneratorOptions = {}) {
    if (!apiKey) {
      throw new Error('OpenAI API key is required for embeddings');
    }

    this.logger = options.logger ?? console;
    this.batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    this.maxConcurrency = options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
    this.cacheSize = MAX_CACHE_ENTRIES;
    this.rateLimitPerMinute = options.rateLimitPerMinute ?? DEFAULT_RATE_LIMIT_PER_MINUTE;

    this.cache = new NodeCache({
      stdTTL: options.cacheTtlSeconds ?? 60 * 60,
      checkperiod: 120,
      useClones: false,
    });

    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: apiKey,
      model: options.model ?? DEFAULT_MODEL,
    });
  }

  /**
   * Generate an embedding for a single text string.
   */
  async embedText(text: string): Promise<number[]> {
    const trimmed = text.trim();
    if (!trimmed) {
      throw new Error('Cannot embed empty text');
    }

    const key = cacheKey(trimmed);
    const cached = this.cache.get<number[]>(key);
    if (cached) {
      return cached;
    }

    const embedding = await this.executeWithRetry(async () => {
      await this.enforceRateLimit();
      const result = await this.embeddings.embedQuery(trimmed);
      return result;
    });

    this.setCacheEntry(key, embedding);
    this.ensureVectorSize(embedding, trimmed);
    return embedding;
  }

  /**
   * Generate embeddings for a list of texts.
   */
  async embedBatch(texts: string[], batchSize = this.batchSize): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    const results: Array<number[] | undefined> = new Array(texts.length);
    const batches = this.createBatches(texts, batchSize);
    let processed = 0;

    await this.runWithConcurrency(batches, async ({ start, items }) => {
      const vectors = await this.embedTextsChunk(items);
      vectors.forEach((vector, idx) => {
        results[start + idx] = vector;
      });
      processed += items.length;
      if (texts.length >= 50 && processed % 50 === 0) {
        this.logger.info?.(`Embedded ${processed}/${texts.length} texts`);
      }
    });

    return results.map((vector, index) => {
      if (!vector) {
        throw new Error(`Missing embedding for text index ${index}`);
      }
      return vector;
    });
  }

  /**
   * Embed chunk content and attach embeddings to each chunk.
   */
  async embedChunks(chunks: SEBIChunk[]): Promise<SEBIChunk[]> {
    if (chunks.length === 0) {
      return [];
    }

    const embeddings = await this.embedBatch(chunks.map((chunk) => chunk.content));

    return chunks.map((chunk, idx) => {
      const enriched = {
        ...chunk,
        embedding: embeddings[idx],
      } satisfies SEBIChunk;
      return SEBIChunkSchema.parse(enriched);
    });
  }

  private createBatches(items: string[], batchSize: number): Array<{ start: number; items: string[] }> {
    const batches: Array<{ start: number; items: string[] }> = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push({ start: i, items: items.slice(i, i + batchSize) });
    }
    return batches;
  }

  private async runWithConcurrency(
    items: Array<{ start: number; items: string[] }>,
    worker: (item: { start: number; items: string[] }) => Promise<void>,
  ): Promise<void> {
    const queue = [...items];
    const workers: Promise<void>[] = [];
    const concurrency = Math.min(this.maxConcurrency, queue.length);

    const runWorker = async (): Promise<void> => {
      while (queue.length > 0) {
        const next = queue.shift();
        if (!next) break;
        await worker(next);
      }
    };

    for (let i = 0; i < concurrency; i++) {
      workers.push(runWorker());
    }

    await Promise.all(workers);
  }

  private async embedTextsChunk(texts: string[]): Promise<number[][]> {
    const vectors: Array<number[] | undefined> = new Array(texts.length);
    const uncachedTexts: string[] = [];
    const uncachedIndices: number[] = [];

    texts.forEach((text, index) => {
      const key = cacheKey(text);
      const cached = this.cache.get<number[]>(key);
      if (cached) {
        vectors[index] = cached;
        return;
      }
      uncachedTexts.push(text);
      uncachedIndices.push(index);
    });

    if (uncachedTexts.length > 0) {
      const fetched = await this.executeWithRetry(async () => {
        await this.enforceRateLimit();
        const result = await this.embeddings.embedDocuments(uncachedTexts);
        return result;
      });

      fetched.forEach((vector, idx) => {
        this.ensureVectorSize(vector, uncachedTexts[idx]);
        const key = cacheKey(uncachedTexts[idx]);
        this.setCacheEntry(key, vector);
        vectors[uncachedIndices[idx]] = vector;
      });
    }

    return vectors.map((vector, idx) => {
      if (!vector) {
        throw new Error(`Missing embedding for batch entry ${idx}`);
      }
      return vector;
    });
  }

  private setCacheEntry(key: string, value: number[]): void {
    if (this.cache.keys().length >= this.cacheSize && !this.cache.has(key)) {
      const oldestKey = this.cache.keys()[0];
      if (oldestKey) {
        this.cache.del(oldestKey);
      }
    }
    this.cache.set(key, value);
  }

  private ensureVectorSize(vector: number[], sourceText: string): void {
    if (vector.length !== VECTOR_SIZE) {
      this.logger.warn?.(
        `Unexpected embedding size (${vector.length}) for text snippet: ${sourceText.slice(0, 32)}...`,
      );
    }
  }

  private async executeWithRetry<T>(fn: () => Promise<T>, attempts = 5): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        const delayMs = 2 ** attempt * 100;
        this.logger.warn?.(`Embedding request failed (attempt ${attempt}): ${this.stringifyError(error)}`);
        await delay(delayMs);
      }
    }

    throw lastError;
  }

  private async enforceRateLimit(): Promise<void> {
    const now = Date.now();
    // Remove timestamps older than 1 minute
    while (this.requestTimestamps.length > 0 && now - this.requestTimestamps[0] > ONE_MINUTE_MS) {
      this.requestTimestamps.shift();
    }

    if (this.requestTimestamps.length >= this.rateLimitPerMinute) {
      const waitMs = ONE_MINUTE_MS - (now - this.requestTimestamps[0]) + 5;
      await delay(waitMs);
    }

    this.requestTimestamps.push(Date.now());
  }

  private stringifyError(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return JSON.stringify(error);
  }
}
