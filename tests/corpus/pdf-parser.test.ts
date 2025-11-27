import { describe, expect, it } from 'vitest';

import { chunkDocument, countTokens, extractSections } from '../../src/corpus/pdf-parser.js';
import { type SEBIDocument } from '../../src/types/sebi-document.js';

// ─────────────────────────────────────────────────────────────────────────────
// Sample Fixtures
// ─────────────────────────────────────────────────────────────────────────────

const SAMPLE_DOCUMENT_TEXT = `
SEBI/HO/IMD/IMD-II/DOF3/P/CIR/2024/001

Date: January 15, 2024

Subject: Amendments to TER Structure for Mutual Funds

Chapter 1: Introduction

1.1 Background

SEBI has been reviewing the Total Expense Ratio (TER) structure applicable to mutual funds. 
This circular introduces amendments aimed at reducing investor costs while maintaining 
operational viability for Asset Management Companies (AMCs).

1.2 Scope

This circular is applicable to all mutual fund schemes regulated by SEBI.

Chapter 2: TER Amendments

2.1 New TER Limits

The following table outlines the revised TER limits based on AUM:

2.1.1 Equity Schemes

For equity oriented schemes, the maximum TER shall be as follows based on daily net assets.

2.1.2 Debt Schemes

For debt oriented schemes, lower TER limits apply to protect investor returns.

2.2 Implementation Timeline

All AMCs must implement the revised TER structure within 90 days from the date of this circular.

Chapter 3: Compliance Requirements

3.1 Reporting

AMCs shall submit monthly compliance reports to SEBI detailing TER calculations for each scheme.
`;

const SIMPLE_TEXT = 'Hello world. This is a test. Testing token counting functionality.';

// ─────────────────────────────────────────────────────────────────────────────
// Token Counting Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('countTokens', () => {
  it('should count tokens in simple text', () => {
    const count = countTokens(SIMPLE_TEXT);
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(50);
  });

  it('should return 0 for empty string', () => {
    const count = countTokens('');
    expect(count).toBe(0);
  });

  it('should handle special characters', () => {
    const count = countTokens('Hello! @#$%^&*() World');
    expect(count).toBeGreaterThan(0);
  });

  it('should handle long text', () => {
    const longText = 'word '.repeat(1000);
    const count = countTokens(longText);
    expect(count).toBeGreaterThan(500);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Section Extraction Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('extractSections', () => {
  it('should extract chapters from document', () => {
    const sections = extractSections(SAMPLE_DOCUMENT_TEXT);

    expect(sections.length).toBeGreaterThan(0);

    const chapter1 = sections.find((s) => s.id === '1');
    expect(chapter1).toBeDefined();
    expect(chapter1?.title).toBe('Introduction');
    expect(chapter1?.level).toBe(1);
  });

  it('should extract nested sections', () => {
    const sections = extractSections(SAMPLE_DOCUMENT_TEXT);

    const chapter1 = sections.find((s) => s.id === '1');
    expect(chapter1?.children.length).toBeGreaterThan(0);

    const section11 = chapter1?.children.find((s) => s.id === '1.1');
    expect(section11).toBeDefined();
    expect(section11?.title).toBe('Background');
    expect(section11?.level).toBe(2);
  });

  it('should extract sub-sections', () => {
    const sections = extractSections(SAMPLE_DOCUMENT_TEXT);

    const chapter2 = sections.find((s) => s.id === '2');
    const section21 = chapter2?.children.find((s) => s.id === '2.1');
    expect(section21?.children.length).toBeGreaterThan(0);

    const subSection211 = section21?.children.find((s) => s.id === '2.1.1');
    expect(subSection211).toBeDefined();
    expect(subSection211?.title).toBe('Equity Schemes');
    expect(subSection211?.level).toBe(3);
  });

  it('should handle text without sections', () => {
    const plainText = 'This is just plain text without any section markers.';
    const sections = extractSections(plainText);
    expect(sections).toEqual([]);
  });

  it('should preserve content within sections', () => {
    const sections = extractSections(SAMPLE_DOCUMENT_TEXT);

    const chapter1 = sections.find((s) => s.id === '1');
    const section11 = chapter1?.children.find((s) => s.id === '1.1');

    expect(section11?.content).toContain('SEBI has been reviewing');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Document Chunking Tests
// ─────────────────────────────────────────────────────────────────────────────

describe('chunkDocument', () => {
  const createTestDocument = (content: string): SEBIDocument => ({
    id: 'test-doc-001',
    circular_id: 'SEBI/HO/TEST/2024/001',
    title: 'Test Document',
    date: new Date('2024-01-15'),
    category: 'mutual_funds',
    content,
    url: 'https://www.sebi.gov.in/test',
    metadata: {},
  });

  it('should create chunks from document with sections', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].document_id).toBe('test-doc-001');
  });

  it('should assign sequential chunk indices', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    chunks.forEach((chunk, index) => {
      expect(chunk.chunk_index).toBe(index);
    });
  });

  it('should generate unique chunk IDs', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    const chunkIds = chunks.map((c) => c.chunk_id);
    const uniqueIds = new Set(chunkIds);

    expect(uniqueIds.size).toBe(chunkIds.length);
  });

  it('should respect maxTokens limit', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc, { maxTokens: 100 });

    for (const chunk of chunks) {
      // Allow some tolerance for sentence boundaries
      expect(chunk.tokens).toBeLessThanOrEqual(150);
    }
  });

  it('should include section hierarchy', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    const chunksWithHierarchy = chunks.filter((c) => c.section_hierarchy.length > 0);
    expect(chunksWithHierarchy.length).toBeGreaterThan(0);
  });

  it('should handle document without sections', () => {
    const plainContent = 'This is plain text. '.repeat(50);
    const doc = createTestDocument(plainContent);
    const chunks = chunkDocument(doc);

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].section_hierarchy).toEqual([]);
  });

  it('should count tokens accurately in chunks', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    for (const chunk of chunks) {
      const actualTokens = countTokens(chunk.content);
      expect(chunk.tokens).toBe(actualTokens);
    }
  });

  it('should use default options when not specified', () => {
    const doc = createTestDocument(SAMPLE_DOCUMENT_TEXT);
    const chunks = chunkDocument(doc);

    // Default maxTokens is 512, chunks should not exceed this significantly
    for (const chunk of chunks) {
      expect(chunk.tokens).toBeLessThanOrEqual(600); // Some tolerance
    }
  });
});
