import pdfParse from 'pdf-parse';
import { get_encoding } from 'tiktoken';

import {
  type SEBICategory,
  type SEBIChunk,
  type SEBIDocument,
  SEBIChunkSchema,
  SEBIDocumentSchema,
} from '../types/sebi-document.js';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Represents a parsed section of a SEBI document.
 */
export interface Section {
  /** Section identifier (e.g., "2.3.1"). */
  id: string;
  /** Section title. */
  title: string;
  /** Section content (excluding nested sections). */
  content: string;
  /** Hierarchical depth (1 = chapter, 2 = section, 3 = sub-section). */
  level: number;
  /** Child sections. */
  children: Section[];
}

/**
 * Options for chunking a document.
 */
export interface ChunkOptions {
  /** Maximum tokens per chunk (default: 512). */
  maxTokens?: number;
  /** Minimum tokens per chunk (default: 128). */
  minTokens?: number;
  /** Token overlap between adjacent chunks (default: 50). */
  overlap?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Counting
// ─────────────────────────────────────────────────────────────────────────────

/** Cached tiktoken encoder for cl100k_base. */
let encoder: ReturnType<typeof get_encoding> | null = null;

/**
 * Get or initialize the tiktoken encoder.
 */
function getEncoder(): ReturnType<typeof get_encoding> {
  if (!encoder) {
    encoder = get_encoding('cl100k_base');
  }
  return encoder;
}

/**
 * Count tokens in a string using tiktoken cl100k_base encoding.
 *
 * @param text - The text to count tokens for.
 * @returns The number of tokens.
 *
 * @example
 * ```ts
 * const count = countTokens('Hello, world!');
 * console.log(count); // e.g., 4
 * ```
 */
export function countTokens(text: string): number {
  const enc = getEncoder();
  const tokens = enc.encode(text);
  return tokens.length;
}

// ─────────────────────────────────────────────────────────────────────────────
// Section Extraction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Regex patterns for SEBI document structure.
 */
const SECTION_PATTERNS = {
  /** Matches "Chapter 1: Title" or "Chapter I. Title" */
  chapter: /^(?:Chapter)\s+(\d+|[IVXLCDM]+)[:\.\s]+([^\n]+)/gim,
  /** Matches "1.2 Title" or "1.2: Title" */
  section: /^(\d+\.\d+)[:\.\s]+([^\n]+)/gm,
  /** Matches "1.2.3 Title" or "1.2.3: Title" */
  subSection: /^(\d+\.\d+\.\d+)[:\.\s]+([^\n]+)/gm,
  /** Matches "(a) Title" or "(i) Title" for alphabetic/roman sub-items */
  alphabetic: /^\(([a-z]|[ivxlcdm]+)\)[:\.\s]+([^\n]+)/gim,
};

/**
 * Metadata extracted from SEBI circular header.
 */
interface CircularMetadata {
  circularId: string | null;
  date: Date | null;
  title: string | null;
}

/**
 * Extract circular metadata from document header.
 *
 * @param text - Full document text.
 * @returns Extracted metadata.
 */
function extractCircularMetadata(text: string): CircularMetadata {
  const result: CircularMetadata = {
    circularId: null,
    date: null,
    title: null,
  };

  // Pattern for SEBI circular ID: SEBI/HO/XXX/XXX/YYYY/NNN
  const circularIdMatch = text.match(/SEBI\/[A-Z]+\/[A-Z]+(?:\/[A-Z0-9-]+)*\/\d{4}\/\d+/i);
  if (circularIdMatch) {
    result.circularId = circularIdMatch[0];
  }

  // Pattern for date: various formats
  const datePatterns = [
    /(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})/, // DD/MM/YYYY or DD-MM-YYYY
    /(\w+)\s+(\d{1,2}),?\s+(\d{4})/, // Month DD, YYYY
    /(\d{1,2})\s+(\w+)\s+(\d{4})/, // DD Month YYYY
  ];

  for (const pattern of datePatterns) {
    const dateMatch = text.slice(0, 2000).match(pattern);
    if (dateMatch) {
      const parsed = new Date(dateMatch[0]);
      if (!isNaN(parsed.getTime())) {
        result.date = parsed;
        break;
      }
    }
  }

  // Extract title from first significant line
  const lines = text.split('\n').filter((line) => line.trim().length > 10);
  if (lines.length > 0) {
    // Skip lines that look like headers/addresses
    for (const line of lines.slice(0, 10)) {
      const trimmed = line.trim();
      if (
        !trimmed.match(/^(SEBI|Securities|To:|From:|Date:|Ref:|Subject:)/i) &&
        trimmed.length > 20 &&
        trimmed.length < 200
      ) {
        result.title = trimmed;
        break;
      }
    }
  }

  return result;
}

/**
 * Detect document category based on content keywords.
 *
 * @param text - Document text.
 * @returns Detected category.
 */
function detectCategory(text: string): SEBICategory {
  const lowerText = text.toLowerCase();

  if (lowerText.includes('mutual fund') || lowerText.includes('amc') || lowerText.includes('ter')) {
    return 'mutual_funds';
  }
  if (lowerText.includes('portfolio manager')) {
    return 'portfolio_managers';
  }
  if (lowerText.includes('invit') || lowerText.includes('infrastructure investment trust')) {
    return 'invits';
  }
  if (lowerText.includes('reit') || lowerText.includes('real estate investment trust')) {
    return 'reits';
  }

  return 'general';
}

/**
 * Extract hierarchical sections from document text.
 *
 * @param text - Full document text.
 * @returns Array of top-level sections with nested children.
 *
 * @example
 * ```ts
 * const sections = extractSections(documentText);
 * console.log(sections[0].title); // "Chapter 1: Introduction"
 * ```
 */
export function extractSections(text: string): Section[] {
  const sections: Section[] = [];
  const lines = text.split('\n');

  let currentChapter: Section | null = null;
  let currentSection: Section | null = null;
  let currentSubSection: Section | null = null;
  let contentBuffer: string[] = [];

  /**
   * Flush accumulated content to the appropriate section.
   */
  function flushContent(): void {
    const content = contentBuffer.join('\n').trim();
    if (!content) return;

    if (currentSubSection) {
      currentSubSection.content += (currentSubSection.content ? '\n' : '') + content;
    } else if (currentSection) {
      currentSection.content += (currentSection.content ? '\n' : '') + content;
    } else if (currentChapter) {
      currentChapter.content += (currentChapter.content ? '\n' : '') + content;
    }
    contentBuffer = [];
  }

  for (const line of lines) {
    const trimmedLine = line.trim();

    // Check for chapter
    const chapterMatch = trimmedLine.match(/^Chapter\s+(\d+|[IVXLCDM]+)[:\.\s]+(.+)/i);
    if (chapterMatch) {
      flushContent();
      currentChapter = {
        id: chapterMatch[1],
        title: chapterMatch[2].trim(),
        content: '',
        level: 1,
        children: [],
      };
      sections.push(currentChapter);
      currentSection = null;
      currentSubSection = null;
      continue;
    }

    // Check for section (e.g., "2.3 Title")
    const sectionMatch = trimmedLine.match(/^(\d+\.\d+)[:\.\s]+(.+)/);
    if (sectionMatch && !trimmedLine.match(/^\d+\.\d+\.\d+/)) {
      flushContent();
      currentSection = {
        id: sectionMatch[1],
        title: sectionMatch[2].trim(),
        content: '',
        level: 2,
        children: [],
      };
      if (currentChapter) {
        currentChapter.children.push(currentSection);
      } else {
        sections.push(currentSection);
      }
      currentSubSection = null;
      continue;
    }

    // Check for sub-section (e.g., "2.3.1 Title")
    const subSectionMatch = trimmedLine.match(/^(\d+\.\d+\.\d+)[:\.\s]+(.+)/);
    if (subSectionMatch) {
      flushContent();
      currentSubSection = {
        id: subSectionMatch[1],
        title: subSectionMatch[2].trim(),
        content: '',
        level: 3,
        children: [],
      };
      if (currentSection) {
        currentSection.children.push(currentSubSection);
      } else if (currentChapter) {
        currentChapter.children.push(currentSubSection);
      } else {
        sections.push(currentSubSection);
      }
      continue;
    }

    // Regular content line
    if (trimmedLine) {
      contentBuffer.push(trimmedLine);
    }
  }

  // Flush any remaining content
  flushContent();

  return sections;
}

/**
 * Build section hierarchy path as string array.
 *
 * @param sections - Flat list of section titles.
 * @returns Hierarchy path array.
 */
function buildHierarchy(sections: string[]): string[] {
  return sections.filter(Boolean);
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF Parsing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Clean extracted PDF text by normalizing whitespace and removing artifacts.
 *
 * @param text - Raw extracted text.
 * @returns Cleaned text.
 */
function cleanPdfText(text: string): string {
  return (
    text
      // Normalize line endings
      .replace(/\r\n/g, '\n')
      .replace(/\r/g, '\n')
      // Remove excessive whitespace
      .replace(/[ \t]+/g, ' ')
      // Remove page numbers and headers (common patterns)
      .replace(/^Page\s+\d+\s*$/gm, '')
      .replace(/^\d+\s*$/gm, '')
      // Fix hyphenated words split across lines
      .replace(/(\w)-\n(\w)/g, '$1$2')
      // Collapse multiple newlines
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  );
}

/**
 * Parse a SEBI PDF document and extract structured data.
 *
 * @param pdfBuffer - Buffer containing PDF file data.
 * @returns Parsed SEBI document.
 * @throws Error if PDF parsing fails.
 *
 * @example
 * ```ts
 * import { readFile } from 'fs/promises';
 *
 * const buffer = await readFile('circular.pdf');
 * const document = await parseSEBIPDF(buffer);
 * console.log(document.circular_id);
 * ```
 */
export async function parseSEBIPDF(pdfBuffer: Buffer): Promise<SEBIDocument> {
  let pdfData;

  try {
    pdfData = await pdfParse(pdfBuffer);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Failed to parse PDF: ${message}`);
  }

  const rawText = pdfData.text;
  const cleanedText = cleanPdfText(rawText);
  const metadata = extractCircularMetadata(cleanedText);
  const category = detectCategory(cleanedText);

  // Generate a unique ID
  const id = metadata.circularId
    ? metadata.circularId.replace(/\//g, '-').toLowerCase()
    : `sebi-doc-${Date.now()}`;

  const document: SEBIDocument = {
    id,
    circular_id: metadata.circularId ?? `UNKNOWN-${Date.now()}`,
    title: metadata.title ?? 'Untitled SEBI Circular',
    date: metadata.date ?? new Date(),
    category,
    content: cleanedText,
    url: '',
    metadata: {
      pageCount: pdfData.numpages,
      pdfInfo: pdfData.info,
      extractedAt: new Date().toISOString(),
    },
  };

  // Validate with Zod schema
  return SEBIDocumentSchema.parse(document);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Split text into sentences for finer-grained chunking.
 *
 * @param text - Text to split.
 * @returns Array of sentences.
 */
function splitIntoSentences(text: string): string[] {
  // Split on sentence boundaries while preserving the delimiter
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Split text into chunks respecting token limits.
 *
 * @param text - Text to split.
 * @param maxTokens - Maximum tokens per chunk.
 * @param overlap - Token overlap between chunks.
 * @returns Array of text chunks.
 */
function splitByTokens(text: string, maxTokens: number, overlap: number): string[] {
  const sentences = splitIntoSentences(text);
  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentTokens = 0;

  for (const sentence of sentences) {
    const sentenceTokens = countTokens(sentence);

    if (currentTokens + sentenceTokens > maxTokens && currentChunk.length > 0) {
      // Save current chunk
      chunks.push(currentChunk.join(' '));

      // Calculate overlap
      if (overlap > 0) {
        let overlapTokens = 0;
        const overlapSentences: string[] = [];

        // Add sentences from the end until we reach overlap target
        for (let i = currentChunk.length - 1; i >= 0 && overlapTokens < overlap; i--) {
          const sent = currentChunk[i];
          overlapSentences.unshift(sent);
          overlapTokens += countTokens(sent);
        }

        currentChunk = overlapSentences;
        currentTokens = overlapTokens;
      } else {
        currentChunk = [];
        currentTokens = 0;
      }
    }

    currentChunk.push(sentence);
    currentTokens += sentenceTokens;
  }

  // Add remaining content
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(' '));
  }

  return chunks;
}

/**
 * Flatten section hierarchy into a list of content blocks with paths.
 */
interface ContentBlock {
  content: string;
  hierarchy: string[];
}

/**
 * Recursively flatten sections into content blocks.
 *
 * @param sections - Hierarchical sections.
 * @param parentPath - Parent hierarchy path.
 * @returns Flat list of content blocks.
 */
function flattenSections(sections: Section[], parentPath: string[] = []): ContentBlock[] {
  const blocks: ContentBlock[] = [];

  for (const section of sections) {
    const currentPath = [...parentPath, `${section.id} ${section.title}`.trim()];

    if (section.content) {
      blocks.push({
        content: section.content,
        hierarchy: currentPath,
      });
    }

    if (section.children.length > 0) {
      blocks.push(...flattenSections(section.children, currentPath));
    }
  }

  return blocks;
}

/**
 * Chunk a SEBI document into smaller pieces suitable for embedding.
 *
 * @param document - The document to chunk.
 * @param options - Chunking options.
 * @returns Array of document chunks.
 *
 * @example
 * ```ts
 * const chunks = chunkDocument(document, { maxTokens: 512, overlap: 50 });
 * console.log(chunks.length);
 * ```
 */
export function chunkDocument(document: SEBIDocument, options: ChunkOptions = {}): SEBIChunk[] {
  const { maxTokens = 512, minTokens = 128, overlap = 50 } = options;

  const sections = extractSections(document.content);
  const chunks: SEBIChunk[] = [];
  let chunkIndex = 0;

  // If no sections found, chunk the entire content
  if (sections.length === 0) {
    const textChunks = splitByTokens(document.content, maxTokens, overlap);

    for (const text of textChunks) {
      const tokens = countTokens(text);

      // Skip chunks that are too small (unless it's the only chunk)
      if (tokens < minTokens && textChunks.length > 1) {
        continue;
      }

      const chunk: SEBIChunk = {
        chunk_id: `${document.id}-chunk-${chunkIndex}`,
        document_id: document.id,
        chunk_index: chunkIndex,
        content: text,
        tokens,
        section_hierarchy: [],
      };

      chunks.push(SEBIChunkSchema.parse(chunk));
      chunkIndex++;
    }

    return chunks;
  }

  // Flatten sections and chunk each
  const contentBlocks = flattenSections(sections);

  for (const block of contentBlocks) {
    const blockTokens = countTokens(block.content);

    if (blockTokens <= maxTokens) {
      // Block fits in a single chunk
      if (blockTokens >= minTokens || contentBlocks.length === 1) {
        const chunk: SEBIChunk = {
          chunk_id: `${document.id}-chunk-${chunkIndex}`,
          document_id: document.id,
          chunk_index: chunkIndex,
          content: block.content,
          tokens: blockTokens,
          section_hierarchy: block.hierarchy,
        };

        chunks.push(SEBIChunkSchema.parse(chunk));
        chunkIndex++;
      }
    } else {
      // Block needs to be split
      const textChunks = splitByTokens(block.content, maxTokens, overlap);

      for (const text of textChunks) {
        const tokens = countTokens(text);

        if (tokens >= minTokens || textChunks.length === 1) {
          const chunk: SEBIChunk = {
            chunk_id: `${document.id}-chunk-${chunkIndex}`,
            document_id: document.id,
            chunk_index: chunkIndex,
            content: text,
            tokens,
            section_hierarchy: block.hierarchy,
          };

          chunks.push(SEBIChunkSchema.parse(chunk));
          chunkIndex++;
        }
      }
    }
  }

  return chunks;
}
