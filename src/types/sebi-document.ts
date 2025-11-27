import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Category of a SEBI regulatory document.
 */
export const SEBICategorySchema = z.enum([
  'mutual_funds',
  'portfolio_managers',
  'invits',
  'reits',
  'general',
]);

/** Allowed SEBI document categories. */
export type SEBICategory = z.infer<typeof SEBICategorySchema>;

/**
 * Type of user intent inferred from a query.
 */
export const SEBIIntentTypeSchema = z.enum([
  'TER',
  'NAV',
  'disclosure',
  'categorization',
  'risk',
  'general',
]);

/** Allowed SEBI intent types. */
export type SEBIIntentType = z.infer<typeof SEBIIntentTypeSchema>;

/**
 * Priority level for a compliance alert.
 */
export const AlertPrioritySchema = z.enum(['high', 'medium', 'low']);

/** Allowed priority levels. */
export type AlertPriority = z.infer<typeof AlertPrioritySchema>;

// ─────────────────────────────────────────────────────────────────────────────
// SEBIDocument
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Zod schema for a SEBI regulatory document.
 *
 * @example
 * ```ts
 * const doc = SEBIDocumentSchema.parse({
 *   id: 'doc-001',
 *   circular_id: 'SEBI/HO/IMD/2024/001',
 *   title: 'TER Amendments',
 *   date: new Date('2024-06-01'),
 *   category: 'mutual_funds',
 *   content: 'All AMCs shall ensure...',
 *   url: 'https://www.sebi.gov.in/...',
 *   metadata: {},
 * });
 * ```
 */
export const SEBIDocumentSchema = z.object({
  /** Unique identifier for the document. */
  id: z.string().min(1),

  /** Official SEBI circular identifier (e.g., "SEBI/HO/IMD/2024/xxx"). */
  circular_id: z.string().min(1),

  /** Title of the circular or document. */
  title: z.string().min(1),

  /** Issue date of the circular. */
  date: z.coerce.date(),

  /** Regulatory category. */
  category: SEBICategorySchema,

  /** Optional chapter reference (e.g., "Chapter 2: TER"). */
  chapter: z.string().optional(),

  /** Optional section reference (e.g., "Section 2.3"). */
  section: z.string().optional(),

  /** Text content (or chunk content if split). */
  content: z.string(),

  /** Source URL of the circular. */
  url: z.string().url(),

  /** Arbitrary additional metadata. */
  metadata: z.record(z.unknown()),
});

/**
 * A SEBI regulatory document.
 */
export type SEBIDocument = z.infer<typeof SEBIDocumentSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// SEBIChunk
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Zod schema for a text chunk derived from a SEBI document.
 *
 * Chunks are produced during corpus ingestion and may optionally include an
 * embedding vector for vector search.
 */
export const SEBIChunkSchema = z.object({
  /** Unique chunk identifier. */
  chunk_id: z.string().min(1),

  /** Parent document identifier. */
  document_id: z.string().min(1),

  /** Zero-based index of this chunk within its parent document. */
  chunk_index: z.number().int().nonnegative(),

  /** The textual content of the chunk. */
  content: z.string(),

  /** Approximate token count. */
  tokens: z.number().int().nonnegative(),

  /** Hierarchical path of section headers (e.g., ["Chapter 2", "Section 2.3"]). */
  section_hierarchy: z.array(z.string()),

  /**
   * Optional dense embedding vector (3072 dimensions for text-embedding-3-large).
   */
  embedding: z.array(z.number()).length(3072).optional(),
});

/**
 * A text chunk derived from a SEBI document.
 */
export type SEBIChunk = z.infer<typeof SEBIChunkSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// SEBIIntent
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Zod schema for the inferred intent of a user query.
 *
 * Intent extraction aids routing and retrieval logic.
 */
export const SEBIIntentSchema = z.object({
  /** The classified intent type. */
  type: SEBIIntentTypeSchema,

  /** Named entities extracted from the query (e.g., scheme names, terms). */
  entities: z.array(z.string()),

  /** Optional timeframe qualifier (e.g., "latest", "2024-06-27"). */
  timeframe: z.string().optional(),

  /** Confidence score between 0 and 1. */
  confidence: z.number().min(0).max(1),
});

/**
 * Inferred intent from a user query.
 */
export type SEBIIntent = z.infer<typeof SEBIIntentSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// ComplianceAlert
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Zod schema for a compliance alert derived from SEBI circulars.
 *
 * Alerts summarise regulatory changes and actionable items for users.
 */
export const ComplianceAlertSchema = z.object({
  /** Unique alert identifier. */
  id: z.string().min(1),

  /** Source circular identifier. */
  circular_id: z.string().min(1),

  /** Short title of the alert. */
  title: z.string().min(1),

  /** Brief summary of the regulatory change. */
  summary: z.string(),

  /** List of actions that must be taken. */
  actionable_items: z.array(z.string()),

  /** Scheme types or names affected. */
  affected_schemes: z.array(z.string()),

  /** Priority classification. */
  priority: AlertPrioritySchema,

  /** Optional compliance deadline. */
  deadline: z.coerce.date().optional(),

  /** Timestamp when the alert was created. */
  created_at: z.coerce.date(),
});

/**
 * A compliance alert summarising regulatory changes.
 */
export type ComplianceAlert = z.infer<typeof ComplianceAlertSchema>;
