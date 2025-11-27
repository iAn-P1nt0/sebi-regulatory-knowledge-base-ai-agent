import { beforeEach, describe, expect, it, vi } from 'vitest';

import { EmbeddingGenerator, cacheKey } from '../../src/corpus/embedder.js';

import type { SEBIChunk } from '../../src/types/sebi-document.js';

const EMBEDDING_LENGTH = 3072;

const embedQueryMock = vi.fn<(text: string) => Promise<number[]>>();
const embedDocumentsMock = vi.fn<(texts: string[]) => Promise<number[][]>>();

vi.mock('@langchain/openai', () => ({
  OpenAIEmbeddings: class {
    embedQuery = embedQueryMock;
    embedDocuments = embedDocumentsMock;
  },
}));

const mockVector = (seed: number): number[] => Array.from({ length: EMBEDDING_LENGTH }, (_, idx) => seed + idx * 0.0001);

const createChunk = (id: number, content: string): SEBIChunk => ({
  chunk_id: `chunk-${id}`,
  document_id: `doc-${id}`,
  chunk_index: id,
  content,
  tokens: content.split(' ').length,
  section_hierarchy: ['Chapter 1', '1.1'],
});

describe('EmbeddingGenerator', () => {
  beforeEach(() => {
    embedQueryMock.mockReset();
    embedDocumentsMock.mockReset();
  });

  it('embeds single text and caches result', async () => {
    embedQueryMock.mockResolvedValue(mockVector(1));
    const generator = new EmbeddingGenerator('fake-key', { logger: { info: () => undefined, warn: () => undefined, error: () => undefined } });

    const first = await generator.embedText('Sample text for embedding');
    expect(first).toHaveLength(EMBEDDING_LENGTH);
    expect(embedQueryMock).toHaveBeenCalledTimes(1);

    embedQueryMock.mockClear();
    const second = await generator.embedText('Sample text for embedding');
    expect(second).toEqual(first);
    expect(embedQueryMock).not.toHaveBeenCalled();
  });

  it('batches embeddings with order preservation', async () => {
    embedDocumentsMock.mockImplementation(async (texts: string[]) => texts.map((text) => mockVector(text.length)));
    const generator = new EmbeddingGenerator('fake-key');

    const inputs = ['alpha', 'beta text', 'gamma'];
    const outputs = await generator.embedBatch(inputs, 2);

    expect(outputs).toHaveLength(inputs.length);
    outputs.forEach((vector, idx) => {
      expect(vector[0]).toBe(inputs[idx].length);
    });
    expect(embedDocumentsMock).toHaveBeenCalledTimes(2);
  });

  it('embeds chunks and attaches vectors', async () => {
    embedDocumentsMock.mockImplementation(async (texts: string[]) => texts.map((text) => mockVector(text.length)));
    const generator = new EmbeddingGenerator('fake-key');

    const chunks = [createChunk(1, 'first chunk text'), createChunk(2, 'second chunk text longer')];
    const enriched = await generator.embedChunks(chunks);

    expect(enriched).toHaveLength(2);
    expect(enriched[0].embedding).toBeDefined();
    expect(enriched[1].embedding).toHaveLength(EMBEDDING_LENGTH);
  });

  it('generates deterministic cache keys', () => {
    const keyA = cacheKey('Hello World');
    const keyB = cacheKey('Hello World');
    const keyC = cacheKey('Different');

    expect(keyA).toHaveLength(32);
    expect(keyA).toBe(keyB);
    expect(keyA).not.toBe(keyC);
  });
});
