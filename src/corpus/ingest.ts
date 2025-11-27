import chalk from 'chalk';
import fs from 'node:fs/promises';
import path from 'node:path';

import type { EmbeddingGenerator } from './embedder.js';
import type { QdrantManager } from './qdrant-client.js';
import { chunkDocument, parseSEBIPDF, type ChunkOptions } from './pdf-parser.js';
import {
  type SEBICategory,
  type SEBIChunk,
  type SEBIDocument,
  SEBIDocumentSchema,
} from '../types/sebi-document.js';

const DEFAULT_CHUNK_OPTIONS: Required<ChunkOptions> = {
  maxTokens: 512,
  minTokens: 128,
  overlap: 50,
};

const CATEGORY_KEYS: SEBICategory[] = ['mutual_funds', 'portfolio_managers', 'invits', 'reits', 'general'];

export interface IngestionConfig {
  sourcePath?: string;
  chunkOptions?: ChunkOptions;
  logger?: Pick<typeof console, 'info' | 'warn' | 'error'>;
}

export interface IngestionSummary {
  pdfPath: string;
  circularId: string;
  chunkCount: number;
  embeddingCount: number;
  durationMs: number;
}

export interface DirectoryIngestionSummary {
  totalFiles: number;
  successCount: number;
  failureCount: number;
  summaries: IngestionSummary[];
  failures: Array<{ file: string; error: string }>;
}

export interface CorpusStats {
  totalChunks: number;
  categories: Record<SEBICategory, number>;
  latestIngestion?: string;
}

interface CorpusDependencies {
  parser?: (buffer: Buffer) => Promise<SEBIDocument>;
  embedder: EmbeddingGenerator;
  qdrant: QdrantManager;
}

export class CorpusIngestion {
  private readonly parser: (buffer: Buffer) => Promise<SEBIDocument>;
  private readonly embedder: EmbeddingGenerator;
  private readonly qdrant: QdrantManager;
  private readonly config: IngestionConfig;

  constructor(deps: CorpusDependencies, config: IngestionConfig = {}) {
    this.parser = deps.parser ?? parseSEBIPDF;
    this.embedder = deps.embedder;
    this.qdrant = deps.qdrant;
    this.config = config;
  }

  private get logger(): Pick<typeof console, 'info' | 'warn' | 'error'> {
    return this.config.logger ?? console;
  }

  async ingestPDF(pdfPath: string, metadata?: Partial<SEBIDocument>): Promise<IngestionSummary> {
    const absolutePath = this.resolvePath(pdfPath);
    await this.qdrant.initializeCollection();

    const start = Date.now();
    this.logger.info?.(chalk.cyan(`Ingesting PDF: ${absolutePath}`));

    const buffer = await fs.readFile(absolutePath);
    const document = await this.parser(buffer);

    const merged = this.mergeMetadata(document, metadata);
    this.validateMetadata(merged);

    const chunks = chunkDocument(merged, this.chunkOptions);
    if (chunks.length === 0) {
      throw new Error(`No chunks produced for ${merged.circular_id}`);
    }
    this.logger.info?.(`Chunked document into ${chunks.length} chunks`);

    const embeddedChunks = await this.embedChunks(chunks);
    const embeddings = embeddedChunks.map((chunk) => {
      if (!chunk.embedding) {
        throw new Error(`Missing embedding for chunk ${chunk.chunk_id}`);
      }
      return chunk.embedding;
    });

    const upserted = await this.qdrant.upsertChunks(embeddedChunks, embeddings);

    const summary: IngestionSummary = {
      pdfPath: absolutePath,
      circularId: merged.circular_id,
      chunkCount: embeddedChunks.length,
      embeddingCount: upserted,
      durationMs: Date.now() - start,
    };

    this.logger.info?.(chalk.green(`Ingested ${summary.circularId} (${summary.chunkCount} chunks)`));
    return summary;
  }

  async ingestDirectory(dirPath: string): Promise<DirectoryIngestionSummary> {
    const directory = this.resolvePath(dirPath);
    const pdfFiles = await this.walkPdfFiles(directory);

    if (pdfFiles.length === 0) {
      this.logger.warn?.(chalk.yellow(`No PDF files found under ${directory}`));
    }

    const summaries: IngestionSummary[] = [];
    const failures: Array<{ file: string; error: string }> = [];

    for (const pdf of pdfFiles) {
      try {
        const summary = await this.ingestPDF(pdf);
        summaries.push(summary);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        failures.push({ file: pdf, error: message });
        this.logger.error?.(chalk.red(`Failed to ingest ${pdf}: ${message}`));
      }
    }

    return {
      totalFiles: pdfFiles.length,
      successCount: summaries.length,
      failureCount: failures.length,
      summaries,
      failures,
    };
  }

  async updateCorpus(circularId: string, pdfPath: string): Promise<IngestionSummary> {
    this.logger.info?.(chalk.yellow(`Updating corpus for ${circularId}`));
    await this.qdrant.deleteChunksByCircularId(circularId);
    return this.ingestPDF(pdfPath, {
      circular_id: circularId,
      metadata: { version: new Date().toISOString() },
    });
  }

  async getCorpusStats(): Promise<CorpusStats> {
    const chunks = await this.qdrant.listChunks();
    const categories: Record<SEBICategory, number> = CATEGORY_KEYS.reduce((acc, key) => {
      acc[key] = 0;
      return acc;
    }, {} as Record<SEBICategory, number>);

    let latestTimestamp = 0;

    for (const result of chunks) {
      const category = result.document.category;
      categories[category] = (categories[category] ?? 0) + 1;

      const timestamp = result.document.date instanceof Date ? result.document.date.getTime() : 0;
      if (timestamp > latestTimestamp) {
        latestTimestamp = timestamp;
      }
    }

    return {
      totalChunks: chunks.length,
      categories,
      latestIngestion: latestTimestamp ? new Date(latestTimestamp).toISOString() : undefined,
    };
  }

  validateMetadata(metadata?: Partial<SEBIDocument>): void {
    if (!metadata) {
      throw new Error('Document metadata is required');
    }

    if (!metadata.circular_id) {
      throw new Error('Missing circular_id in metadata');
    }

    if (!metadata.date) {
      throw new Error('Missing date in metadata');
    }

    if (!metadata.category) {
      throw new Error('Missing category in metadata');
    }

    if (!CATEGORY_KEYS.includes(metadata.category)) {
      throw new Error(`Invalid category: ${metadata.category}`);
    }
  }

  private mergeMetadata(document: SEBIDocument, overrides?: Partial<SEBIDocument>): SEBIDocument {
    const merged = {
      ...document,
      ...overrides,
      metadata: {
        ...document.metadata,
        ...overrides?.metadata,
      },
    };

    return SEBIDocumentSchema.parse(merged);
  }

  private resolvePath(inputPath: string): string {
    if (path.isAbsolute(inputPath)) {
      return inputPath;
    }
    if (this.config.sourcePath) {
      return path.resolve(this.config.sourcePath, inputPath);
    }
    return path.resolve(process.cwd(), inputPath);
  }

  private async embedChunks(chunks: SEBIChunk[]): Promise<SEBIChunk[]> {
    const enriched = await this.embedder.embedChunks(chunks);
    return enriched;
  }

  private get chunkOptions(): Required<ChunkOptions> {
    return {
      ...DEFAULT_CHUNK_OPTIONS,
      ...this.config.chunkOptions,
    };
  }

  private async walkPdfFiles(dir: string): Promise<string[]> {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const pdfs: string[] = [];

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        pdfs.push(...(await this.walkPdfFiles(fullPath)));
        continue;
      }

      if (entry.isFile() && entry.name.toLowerCase().endsWith('.pdf')) {
        pdfs.push(fullPath);
      }
    }

    return pdfs;
  }
}
