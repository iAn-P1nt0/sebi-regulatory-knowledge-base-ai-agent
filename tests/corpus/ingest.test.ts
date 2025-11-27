import fs from 'node:fs/promises';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CorpusIngestion } from '../../src/corpus/ingest.js';

import type { EmbeddingGenerator } from '../../src/corpus/embedder.js';
import type { QdrantManager, SearchResult } from '../../src/corpus/qdrant-client.js';
import type { SEBIChunk, SEBIDocument } from '../../src/types/sebi-document.js';

const createDocument = (overrides: Partial<SEBIDocument> = {}): SEBIDocument => ({
  id: overrides.id ?? 'doc-001',
  circular_id: overrides.circular_id ?? 'SEBI/HO/IMD/2024/001',
  title: overrides.title ?? 'Sample Circular',
  date: overrides.date ?? new Date('2024-01-01'),
  category: overrides.category ?? 'mutual_funds',
  content:
    overrides.content ??
    'This is a sample SEBI circular content. It contains multiple sentences to ensure chunking works as expected.',
  url: overrides.url ?? 'https://www.sebi.gov.in/circulars/sample',
  metadata: overrides.metadata ?? {},
});

const createChunk = (id: string, overrides: Partial<SEBIChunk> = {}): SEBIChunk => ({
  chunk_id: `chunk-${id}`,
  document_id: overrides.document_id ?? 'doc-001',
  chunk_index: Number(id),
  content: overrides.content ?? 'Chunk content for testing.',
  tokens: overrides.tokens ?? 32,
  section_hierarchy: overrides.section_hierarchy ?? ['Chapter 1'],
  ...('embedding' in overrides ? { embedding: overrides.embedding } : {}),
});

const createEmbedding = (): number[] => Array.from({ length: 3072 }, (_, index) => index / 3072);

describe('CorpusIngestion', () => {
  let parserMock: ReturnType<typeof vi.fn<(buffer: Buffer) => Promise<SEBIDocument>>>;
  let embedderMock: EmbeddingGenerator;
  let qdrantMock: QdrantManager;
  let embedChunksSpy: ReturnType<typeof vi.fn>;
  let upsertSpy: ReturnType<typeof vi.fn>;
  let deleteSpy: ReturnType<typeof vi.fn>;
  let listChunksSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    parserMock = vi.fn(async () => createDocument());

    embedChunksSpy = vi.fn(async (chunks: SEBIChunk[]) =>
      chunks.map((chunk) => ({
        ...chunk,
        embedding: createEmbedding(),
      })),
    );

    embedderMock = {
      embedChunks: embedChunksSpy,
    } as unknown as EmbeddingGenerator;

    upsertSpy = vi.fn().mockResolvedValue(1);
    deleteSpy = vi.fn().mockResolvedValue(1);
    listChunksSpy = vi.fn().mockResolvedValue([]);

    qdrantMock = {
      initializeCollection: vi.fn().mockResolvedValue(undefined),
      upsertChunks: upsertSpy,
      deleteChunksByCircularId: deleteSpy,
      listChunks: listChunksSpy,
    } as unknown as QdrantManager;

    vi.spyOn(fs, 'readFile').mockResolvedValue(Buffer.from('PDF'));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('ingests a PDF and upserts embeddings', async () => {
    const ingestion = new CorpusIngestion(
      { parser: parserMock, embedder: embedderMock, qdrant: qdrantMock },
      { sourcePath: '/tmp' },
    );

    const summary = await ingestion.ingestPDF('sample.pdf', {
      circular_id: 'SEBI/HO/IMD/2024/999',
      url: 'https://www.sebi.gov.in/circulars/custom',
    });

    expect(parserMock).toHaveBeenCalledTimes(1);
    expect(embedChunksSpy).toHaveBeenCalledTimes(1);
    expect(upsertSpy).toHaveBeenCalledTimes(1);
    expect(summary.circularId).toBe('SEBI/HO/IMD/2024/999');
    expect(summary.chunkCount).toBeGreaterThan(0);
    expect(summary.embeddingCount).toBe(summary.chunkCount);
  });

  it('computes corpus statistics from qdrant results', async () => {
    const documents: SearchResult[] = [
      {
        score: 0.9,
        chunk: createChunk('1'),
        document: createDocument({ category: 'mutual_funds', date: new Date('2024-02-01') }),
      },
      {
        score: 0.8,
        chunk: createChunk('2'),
        document: createDocument({ category: 'reits', date: new Date('2024-03-15') }),
      },
    ];

    listChunksSpy.mockResolvedValue(documents);

    const ingestion = new CorpusIngestion({ parser: parserMock, embedder: embedderMock, qdrant: qdrantMock });
    const stats = await ingestion.getCorpusStats();

    expect(stats.totalChunks).toBe(2);
    expect(stats.categories.mutual_funds).toBe(1);
    expect(stats.categories.reits).toBe(1);
    expect(stats.latestIngestion).toBeDefined();
  });

  it('updates a circular by deleting existing chunks first', async () => {
    const ingestion = new CorpusIngestion({ parser: parserMock, embedder: embedderMock, qdrant: qdrantMock });

    await ingestion.updateCorpus('SEBI/HO/IMD/2024/001', 'latest.pdf');

    expect(deleteSpy).toHaveBeenCalledWith('SEBI/HO/IMD/2024/001');
    expect(upsertSpy).toHaveBeenCalledTimes(1);
  });
});
