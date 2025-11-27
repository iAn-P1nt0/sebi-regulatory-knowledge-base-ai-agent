import { beforeEach, describe, expect, it } from 'vitest';

import { QdrantManager, type SearchFilters } from '../../src/corpus/qdrant-client.js';
import { type SEBIChunk } from '../../src/types/sebi-document.js';

interface MockPoint {
  id: string | number;
  vector: number[];
  payload: Record<string, unknown>;
}

type MockFilterRule =
  | { key: string; match: { value: string } }
  | { key: string; range: { gte?: number; lte?: number } };

class MockQdrantClient {
  private readonly collections = new Map<string, { points: Map<string | number, MockPoint> }>();

  async createCollection(name: string): Promise<void> {
    if (!this.collections.has(name)) {
      this.collections.set(name, { points: new Map() });
    }
  }

  async getCollection(name: string): Promise<{ points: number }> {
    const collection = this.collections.get(name);
    if (!collection) {
      throw new Error('Collection not found');
    }
    return { points: collection.points.size };
  }

  async createPayloadIndex(): Promise<void> {
    // No-op for mock
  }

  async upsert(
    name: string,
    payload: { points: Array<{ id: string | number; vector: number[]; payload: Record<string, unknown> }> },
  ): Promise<void> {
    const collection = this.ensureCollection(name);
    for (const point of payload.points) {
      collection.points.set(point.id, { ...point });
    }
  }

  async search(
    name: string,
    params: {
      vector: number[];
      limit?: number;
      filter?: Record<string, unknown>;
      with_payload?: boolean;
    },
  ): Promise<Array<{ id: string | number; score: number; payload: Record<string, unknown> }>> {
    const collection = this.ensureCollection(name);
    const candidates = Array.from(collection.points.values()).filter((point) =>
      this.matchesFilter(point.payload, params.filter),
    );

    const scored = candidates
      .map((point) => ({
        id: point.id,
        payload: point.payload,
        score: this.cosineSimilarity(params.vector, point.vector),
      }))
      .sort((a, b) => b.score - a.score);

    return scored.slice(0, params.limit ?? 5);
  }

  async scroll(
    name: string,
    params: {
      limit?: number;
      offset?: number;
      with_payload?: boolean;
      filter?: Record<string, unknown>;
    },
  ): Promise<{ points: MockPoint[]; next_page_offset?: number }> {
    const collection = this.ensureCollection(name);
    const filtered = Array.from(collection.points.values()).filter((point) =>
      this.matchesFilter(point.payload, params.filter),
    );

    const start = params.offset ?? 0;
    const end = start + (params.limit ?? 50);
    const page = filtered.slice(start, end);

    const next = end < filtered.length ? end : undefined;
    return { points: page, next_page_offset: next };
  }

  async deleteCollection(name: string): Promise<void> {
    this.collections.delete(name);
  }

  private ensureCollection(name: string) {
    const collection = this.collections.get(name);
    if (!collection) {
      throw new Error('Collection not found');
    }
    return collection;
  }

  private matchesFilter(payload: Record<string, unknown>, filter?: Record<string, unknown>): boolean {
    if (!filter || !('must' in filter) || !Array.isArray(filter.must)) {
      return true;
    }

    return (filter.must as MockFilterRule[]).every((rule) => {
      if ('match' in rule) {
        return payload[rule.key] === rule.match.value;
      }
      if ('range' in rule) {
        const value = typeof payload[rule.key] === 'number' ? (payload[rule.key] as number) : undefined;
        if (value === undefined) return false;
        const gte = rule.range.gte ?? Number.NEGATIVE_INFINITY;
        const lte = rule.range.lte ?? Number.POSITIVE_INFINITY;
        return value >= gte && value <= lte;
      }
      return true;
    });
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const minLength = Math.min(a.length, b.length);
    let dot = 0;
    let aMag = 0;
    let bMag = 0;
    for (let i = 0; i < minLength; i++) {
      dot += a[i] * b[i];
      aMag += a[i] ** 2;
      bMag += b[i] ** 2;
    }
    if (aMag === 0 || bMag === 0) {
      return 0;
    }
    return dot / (Math.sqrt(aMag) * Math.sqrt(bMag));
  }
}

type ManagerClient = ConstructorParameters<typeof QdrantManager>[1];

const createEmbedding = (value: number): number[] => Array.from({ length: 8 }, (_, i) => (i + 1) * value);

const createChunk = (id: string, content: string, overrides: Record<string, unknown> = {}): SEBIChunk => {
  const base: SEBIChunk = {
    chunk_id: `chunk-${id}`,
    document_id: `doc-${id}`,
    chunk_index: Number(id),
    content,
    tokens: content.split(' ').length,
    section_hierarchy: ['Chapter 1', '1.1'],
  };

  return Object.assign(base, overrides);
};

const createMetadata = (overrides: Record<string, unknown> = {}) => ({
  circular_id: 'SEBI/HO/IMD/2024/001',
  category: 'mutual_funds',
  date: '2024-01-15',
  chapter: 'Chapter 1',
  title: 'Sample Circular',
  url: 'https://sebi.gov.in/circular',
  document: {
    id: 'doc-1',
    circular_id: 'SEBI/HO/IMD/2024/001',
    title: 'Sample Circular',
    date: new Date('2024-01-15'),
    category: 'mutual_funds',
    content: 'Sample content',
    url: 'https://sebi.gov.in/circular',
    metadata: {},
  },
  ...overrides,
});

type ChunkWithMetadata = SEBIChunk & { metadata: ReturnType<typeof createMetadata> };

const withMetadata = (chunk: SEBIChunk, metadata: ReturnType<typeof createMetadata>): ChunkWithMetadata =>
  Object.assign(chunk, { metadata });

describe('QdrantManager', () => {
  const config = { url: 'http://localhost:6334', collectionName: 'sebi_test' };
  let mockClient: MockQdrantClient;
  let manager: QdrantManager;

  beforeEach(() => {
    mockClient = new MockQdrantClient();
    manager = new QdrantManager(config, mockClient as unknown as ManagerClient);
  });

  it('initializes collection only once', async () => {
    await manager.initializeCollection();
    await expect(manager.initializeCollection()).resolves.toBeUndefined();
  });

  it('upserts chunks and returns success count', async () => {
    await manager.initializeCollection();
    const chunk = withMetadata(createChunk('1', 'Regulatory guidance on TER'), createMetadata());

    const count = await manager.upsertChunks([chunk], [createEmbedding(0.5)]);
    expect(count).toBe(1);
  });

  it('performs semantic search with filters', async () => {
    await manager.initializeCollection();
    const chunk = withMetadata(
      createChunk('1', 'TER rules for mutual funds'),
      createMetadata({ category: 'mutual_funds' }),
    );
    await manager.upsertChunks([chunk], [createEmbedding(1)]);

    const filters: SearchFilters = { category: 'mutual_funds' };
    const results = await manager.semanticSearch(createEmbedding(1), filters, 3);

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].document.circular_id).toBe('SEBI/HO/IMD/2024/001');
  });

  it('combines semantic and keyword results via hybrid search', async () => {
    await manager.initializeCollection();
    const chunkA = withMetadata(
      createChunk('1', 'TER limits for equity schemes'),
      createMetadata(),
    );
    const chunkB = withMetadata(
      createChunk('2', 'Disclosure requirements for NAV along with TER'),
      createMetadata({ circular_id: 'SEBI/HO/IMD/2024/002' }),
    );

    await manager.upsertChunks([chunkA, chunkB], [createEmbedding(1), createEmbedding(0.8)]);

    const results = await manager.hybridSearch('TER equity', createEmbedding(1), 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].chunk.content).toContain('TER');
  });

  it('fetches and deletes collection info', async () => {
    await manager.initializeCollection();
    await manager.deleteCollection();
    await expect(manager.getCollectionInfo()).rejects.toThrow();
  });
});
