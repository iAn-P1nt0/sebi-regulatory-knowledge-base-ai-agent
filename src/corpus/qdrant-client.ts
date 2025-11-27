import { QdrantClient } from '@qdrant/js-client-rest';

import {
  type SEBICategory,
  type SEBIChunk,
  type SEBIDocument,
  SEBIChunkSchema,
  SEBIDocumentSchema,
} from '../types/sebi-document.js';

export interface QdrantConfig {
  url: string;
  apiKey?: string;
  collectionName?: string;
  logger?: Pick<typeof console, 'info' | 'warn' | 'error'>;
}

export interface SearchFilters {
  category?: SEBICategory;
  dateFrom?: Date;
  dateTo?: Date;
  chapter?: string;
}

export interface SearchResult {
  score: number;
  chunk: SEBIChunk;
  document: SEBIDocument;
}

interface ChunkMetadata {
  circular_id?: string;
  category?: SEBICategory;
  date?: string | Date;
  chapter?: string;
  title?: string;
  url?: string;
  document?: Partial<SEBIDocument>;
  metadata?: Record<string, unknown>;
}

interface PayloadDocumentShape {
  id?: string;
  circular_id?: string;
  title?: string;
  date?: string;
  category?: SEBICategory;
  chapter?: string;
  section?: string;
  content?: string;
  url?: string;
  metadata?: Record<string, unknown>;
}

interface PayloadShape {
  chunk?: SEBIChunk;
  document?: PayloadDocumentShape;
  circular_id?: string;
  category?: SEBICategory;
  date?: string;
  chapter?: string;
  url?: string;
  title?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

interface QdrantFilter {
  must?: Array<
    | {
        key: string;
        match: { value: string };
      }
    | {
        key: string;
        range: { gte?: number; lte?: number };
      }
  >;
}

interface QdrantPoint {
  id: string | number;
  vector: number[];
  payload?: PayloadShape;
}

const DEFAULT_COLLECTION = 'sebi_regulations';
const VECTOR_SIZE = 3072;
const DEFAULT_BATCH_SIZE = 100;
const RRF_K = 60;

/**
 * Manages Qdrant vector database interactions for SEBI content.
 */
export class QdrantManager {
  private readonly client: QdrantClient;
  private readonly collectionName: string;
  private readonly logger: Pick<typeof console, 'info' | 'warn' | 'error'>;

  constructor(private readonly config: QdrantConfig, client?: QdrantClient) {
    if (!config.url) {
      throw new Error('Qdrant URL is required');
    }

    this.collectionName = config.collectionName ?? DEFAULT_COLLECTION;
    this.logger = config.logger ?? console;
    this.client = client ?? new QdrantClient({ url: config.url, apiKey: config.apiKey });
  }

  /**
   * Initialize the Qdrant collection if it does not already exist.
   */
  async initializeCollection(): Promise<void> {
    const exists = await this.collectionExists();
    if (exists) {
      return;
    }

    await this.client.createCollection(this.collectionName, {
      vectors: {
        size: VECTOR_SIZE,
        distance: 'Cosine',
        on_disk: true,
      },
      hnsw_config: {
        m: 16,
        ef_construct: 100,
      },
      quantization_config: {
        scalar: {
          type: 'int8',
          quantile: 0.99,
          always_ram: false,
        },
      },
    });

    await Promise.all([
      this.createKeywordIndex('circular_id'),
      this.createKeywordIndex('category'),
      this.createKeywordIndex('chapter'),
      this.createDatetimeIndex('date'),
    ]);
  }

  /**
   * Upsert document chunks and embeddings into Qdrant in batches.
   */
  async upsertChunks(chunks: SEBIChunk[], embeddings: number[][]): Promise<number> {
    if (chunks.length !== embeddings.length) {
      throw new Error('Chunks and embeddings length mismatch');
    }

    let success = 0;

    for (let i = 0; i < chunks.length; i += DEFAULT_BATCH_SIZE) {
      const chunkBatch = chunks.slice(i, i + DEFAULT_BATCH_SIZE);
      const embeddingBatch = embeddings.slice(i, i + DEFAULT_BATCH_SIZE);

      await this.retry(async () => {
        await this.client.upsert(this.collectionName, {
          wait: true,
          points: chunkBatch.map((chunk, index) => {
            const embedding = embeddingBatch[index];
            const payload = this.buildPayload(chunk);

            return {
              id: chunk.chunk_id,
              vector: embedding,
              payload,
            } satisfies QdrantPoint;
          }),
        });
      });

      success += chunkBatch.length;
    }

    return success;
  }

  /**
   * Execute a semantic search using vector similarity and optional filters.
   */
  async semanticSearch(
    embedding: number[],
    filters?: SearchFilters,
    limit = 5,
  ): Promise<SearchResult[]> {
    const filter = this.buildFilter(filters);

    const hits = await this.client.search(this.collectionName, {
      vector: embedding,
      limit,
      with_payload: true,
      filter,
    });

    return hits
      .map((hit) => this.mapHitToResult(hit))
      .filter((result): result is SearchResult => Boolean(result));
  }

  /**
   * Execute hybrid search by fusing semantic and keyword scores via RRF.
   */
  async hybridSearch(
    query: string,
    semanticEmbedding: number[],
    limit = 5,
    filters?: SearchFilters,
  ): Promise<SearchResult[]> {
    const semanticHits = await this.semanticSearch(semanticEmbedding, filters, Math.max(limit * 2, 10));
    const keywordHits = await this.keywordSearch(query, Math.max(limit * 2, 10), filters);

    const fused = this.reciprocalRankFusion([semanticHits, keywordHits], limit);
    return fused;
  }

  /**
   * Fetch collection statistics from Qdrant.
   */
  async getCollectionInfo(): Promise<unknown> {
    return this.client.getCollection(this.collectionName);
  }

  /**
   * Delete the configured collection (useful for tests).
   */
  async deleteCollection(): Promise<void> {
    await this.client.deleteCollection(this.collectionName);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Internal helpers
  // ───────────────────────────────────────────────────────────────────────────

  private async collectionExists(): Promise<boolean> {
    try {
      await this.client.getCollection(this.collectionName);
      return true;
    } catch (error) {
      this.logger.warn(`Collection ${this.collectionName} not found: ${this.stringifyError(error)}`);
      return false;
    }
  }

  private async createKeywordIndex(field: string): Promise<void> {
    await this.client.createPayloadIndex(this.collectionName, {
      field_name: field,
      field_schema: 'keyword',
    });
  }

  private async createDatetimeIndex(field: string): Promise<void> {
    await this.client.createPayloadIndex(this.collectionName, {
      field_name: field,
      field_schema: 'datetime',
    });
  }

  private extractChunkMetadata(chunk: SEBIChunk): ChunkMetadata {
    const meta = (chunk as unknown as { metadata?: ChunkMetadata }).metadata ?? {};
    return meta;
  }

  private normalizeDocumentPayload(document?: Partial<SEBIDocument>): PayloadDocumentShape | undefined {
    if (!document) {
      return undefined;
    }

    return {
      id: document.id,
      circular_id: document.circular_id,
      title: document.title,
      date: document.date ? new Date(document.date).toISOString() : undefined,
      category: document.category,
      chapter: document.chapter,
      section: document.section,
      content: document.content,
      url: document.url,
      metadata: document.metadata,
    } satisfies PayloadDocumentShape;
  }

  private buildPayload(chunk: SEBIChunk): PayloadShape {
    const meta = this.extractChunkMetadata(chunk);
    const chapter = meta.chapter ?? chunk.section_hierarchy[0] ?? null;
    const dateValue = meta.date ? new Date(meta.date).toISOString() : null;

    return {
      chunk,
      document: this.normalizeDocumentPayload(meta.document),
      circular_id: meta.circular_id ?? chunk.document_id,
      category: meta.category ?? 'general',
      date: dateValue ?? undefined,
      chapter: chapter ?? undefined,
      url: meta.url ?? undefined,
      title: meta.title ?? undefined,
      metadata: meta.metadata ?? {},
    } satisfies PayloadShape;
  }

  private buildFilter(filters?: SearchFilters): QdrantFilter | undefined {
    if (!filters) {
      return undefined;
    }

    const must: NonNullable<QdrantFilter['must']> = [];

    if (filters.category) {
      must.push({
        key: 'category',
        match: { value: filters.category },
      });
    }

    if (filters.chapter) {
      must.push({
        key: 'chapter',
        match: { value: filters.chapter },
      });
    }

    if (filters.dateFrom || filters.dateTo) {
      must.push({
        key: 'date',
        range: {
          gte: filters.dateFrom ? filters.dateFrom.getTime() : undefined,
          lte: filters.dateTo ? filters.dateTo.getTime() : undefined,
        },
      });
    }

    if (must.length === 0) {
      return undefined;
    }

    return { must } satisfies QdrantFilter;
  }

  private mapHitToResult(
    hit: { id: string | number; score?: number; payload?: PayloadShape | null },
  ): SearchResult | null {
    const payload = hit.payload;
    if (!payload || !payload.chunk) {
      return null;
    }

    const chunk = SEBIChunkSchema.safeParse(payload.chunk);
    if (!chunk.success) {
      this.logger.warn('Failed to parse chunk payload', chunk.error);
      return null;
    }

    const document = this.payloadToDocument(payload, chunk.data);
    return {
      score: hit.score ?? 0,
      chunk: chunk.data,
      document,
    };
  }

  private payloadToDocument(payload: PayloadShape, chunk: SEBIChunk): SEBIDocument {
    const rawDocument = payload.document ?? {};
    const dateSource = rawDocument.date ?? payload.date ?? new Date().toISOString();
    const parsedDate = new Date(dateSource);
    const safeDate = Number.isNaN(parsedDate.getTime()) ? new Date() : parsedDate;

    const docCandidate = {
      id: rawDocument.id ?? chunk.document_id,
      circular_id: rawDocument.circular_id ?? payload.circular_id ?? chunk.document_id,
      title: rawDocument.title ?? payload.title ?? `SEBI Circular ${chunk.document_id}`,
      date: safeDate,
      category: rawDocument.category ?? payload.category ?? ('general' as SEBICategory),
      chapter: rawDocument.chapter ?? payload.chapter ?? chunk.section_hierarchy[0],
      section: rawDocument.section ?? chunk.section_hierarchy.at(-1),
      content: rawDocument.content ?? chunk.content,
      url: rawDocument.url ?? payload.url ?? '',
      metadata: rawDocument.metadata ?? payload.metadata ?? {},
    } satisfies SEBIDocument;

    return SEBIDocumentSchema.parse(docCandidate);
  }

  private async keywordSearch(
    query: string,
    limit: number,
    filters?: SearchFilters,
  ): Promise<SearchResult[]> {
    const candidates = await this.fetchAllPoints(filters);
    if (!query.trim()) {
      return candidates.slice(0, limit);
    }

    const queryTokens = this.tokenize(query);
    const documents = candidates.map((candidate) => ({
      ...candidate,
      tokens: this.tokenize(candidate.chunk.content),
    }));

    const avgDocLength =
      documents.reduce((sum, doc) => sum + doc.tokens.length, 0) / Math.max(documents.length, 1);

    const docFrequencies = this.computeDocumentFrequency(documents, queryTokens);

    const scored = documents.map((doc) => ({
      ...doc,
      score: this.computeBm25Score(doc.tokens, queryTokens, docFrequencies, documents.length, avgDocLength),
    }));

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((doc) => ({ score: doc.score, chunk: doc.chunk, document: doc.document }));
  }

  private async fetchAllPoints(filters?: SearchFilters): Promise<SearchResult[]> {
    const results: SearchResult[] = [];
    let offset: string | number | Record<string, unknown> | null | undefined = undefined;
    const filter = this.buildFilter(filters);

    while (true) {
      const response = await this.client.scroll(this.collectionName, {
        with_payload: true,
        filter,
        limit: 256,
        offset,
      });

      for (const point of response.points as QdrantPoint[]) {
        const mapped = this.mapHitToResult(point);
        if (mapped) {
          results.push(mapped);
        }
      }

      if (!response.next_page_offset) {
        break;
      }

      offset = response.next_page_offset as string | number | Record<string, unknown> | null | undefined;
    }

    return results;
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .split(/[^a-z0-9]+/g)
      .filter((token) => token.length > 0);
  }

  private computeDocumentFrequency(
    documents: Array<{ tokens: string[] }>,
    queryTokens: string[],
  ): Map<string, number> {
    const df = new Map<string, number>();

    for (const term of queryTokens) {
      let count = 0;
      for (const doc of documents) {
        if (doc.tokens.includes(term)) {
          count += 1;
        }
      }
      df.set(term, count);
    }

    return df;
  }

  private computeBm25Score(
    docTokens: string[],
    queryTokens: string[],
    df: Map<string, number>,
    totalDocs: number,
    avgDocLength: number,
  ): number {
    if (docTokens.length === 0 || queryTokens.length === 0) {
      return 0;
    }

    const k1 = 1.5;
    const b = 0.75;
    const docLength = docTokens.length;
    const tf = new Map<string, number>();

    for (const token of docTokens) {
      tf.set(token, (tf.get(token) ?? 0) + 1);
    }

    let score = 0;

    for (const term of queryTokens) {
      const termFrequency = tf.get(term) ?? 0;
      if (termFrequency === 0) {
        continue;
      }

      const documentFrequency = df.get(term) ?? 0.5;
      const idf = Math.log((totalDocs - documentFrequency + 0.5) / (documentFrequency + 0.5) + 1);
      const numerator = termFrequency * (k1 + 1);
      const denominator = termFrequency + k1 * (1 - b + (b * docLength) / (avgDocLength || 1));
      score += idf * (numerator / denominator);
    }

    return score;
  }

  private reciprocalRankFusion(resultSets: SearchResult[][], limit: number): SearchResult[] {
    const scores = new Map<string, { result: SearchResult; score: number }>();

    resultSets.forEach((results) => {
      results.forEach((result, index) => {
        const key = `${result.document.id}-${result.chunk.chunk_id}`;
        const existing = scores.get(key);
        const increment = 1 / (RRF_K + index + 1);

        if (existing) {
          existing.score += increment;
        } else {
          scores.set(key, { result, score: increment });
        }
      });
    });

    return Array.from(scores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((entry) => ({ ...entry.result, score: entry.score }));
  }

  private async retry<T>(operation: () => Promise<T>, attempts = 3): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        this.logger.warn(
          `Qdrant operation failed (attempt ${attempt}/${attempts}): ${this.stringifyError(error)}`,
        );
        await new Promise((resolve) => setTimeout(resolve, 100 * attempt));
      }
    }

    throw lastError;
  }

  private stringifyError(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return JSON.stringify(error);
  }
}
