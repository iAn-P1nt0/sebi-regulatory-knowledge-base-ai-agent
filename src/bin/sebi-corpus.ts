#!/usr/bin/env node
import 'dotenv/config';

import chalk from 'chalk';
import { Command } from 'commander';
import { createRequire } from 'node:module';
import ora from 'ora';

import { EmbeddingGenerator } from '../corpus/embedder.js';
import { CorpusIngestion, type IngestionSummary } from '../corpus/ingest.js';
import { QdrantManager } from '../corpus/qdrant-client.js';

import type { SEBICategory, SEBIDocument } from '../types/sebi-document.js';

const require = createRequire(import.meta.url);
const pkg = require('../../package.json') as { version?: string };

const CATEGORY_OPTIONS: SEBICategory[] = ['mutual_funds', 'portfolio_managers', 'invits', 'reits', 'general'];

const program = new Command();
program
  .name('sebi-corpus')
  .description('Manage SEBI regulatory corpus ingestion')
  .version(pkg.version ?? '0.0.0');

program
  .command('ingest-file')
  .argument('<pdf>', 'Path to a SEBI circular PDF')
  .option('--circular-id <id>', 'Override circular identifier')
  .option('--category <category>', 'Regulatory category')
  .option('--title <title>', 'Document title override')
  .option('--date <date>', 'Issue date (YYYY-MM-DD)')
  .option('--url <url>', 'Source URL override')
  .description('Ingest a single PDF into the configured Qdrant collection')
  .action(async (pdf: string, options: FileOptions) => {
    const ingestion = createIngestion();
    const metadata = buildMetadataOverrides(options);
    const summary = await runWithSpinner(`Ingesting ${pdf}`, () => ingestion.ingestPDF(pdf, metadata));

    printSummary(summary);
  });

program
  .command('ingest-dir')
  .argument('<directory>', 'Directory tree containing SEBI PDFs')
  .option('--source <path>', 'Override the base source path for relative files')
  .description('Batch-ingest every PDF found under the provided directory (recursively)')
  .action(async (directory: string, options: { source?: string }) => {
    const ingestion = createIngestion({ sourcePath: options.source });
    const summary = await runWithSpinner(`Ingesting directory ${directory}`, () => ingestion.ingestDirectory(directory));

    console.log(chalk.green(`✔ Ingested ${summary.successCount}/${summary.totalFiles} files`));
    if (summary.failureCount > 0) {
      console.error(chalk.red(`✖ ${summary.failureCount} files failed to ingest:`));
      summary.failures.forEach((failure) => {
        console.error(`  - ${failure.file}: ${failure.error}`);
      });
    }
  });

program
  .command('update')
  .argument('<circularId>', 'Circular identifier to replace')
  .argument('<pdf>', 'Updated PDF path')
  .description('Delete and re-ingest chunks for a specific circular')
  .action(async (circularId: string, pdf: string) => {
    const ingestion = createIngestion();
    const summary = await runWithSpinner(`Updating ${circularId}`, () => ingestion.updateCorpus(circularId, pdf));
    printSummary(summary);
  });

program
  .command('stats')
  .description('Display collection statistics from Qdrant')
  .action(async () => {
    const ingestion = createIngestion();
    const stats = await runWithSpinner('Fetching corpus stats', () => ingestion.getCorpusStats());

    console.log(chalk.cyan(`Total chunks: ${stats.totalChunks}`));
    Object.entries(stats.categories).forEach(([category, count]) => {
      console.log(`  • ${category}: ${count}`);
    });
    if (stats.latestIngestion) {
      console.log(`Latest ingestion: ${stats.latestIngestion}`);
    }
  });

interface FileOptions {
  circularId?: string;
  category?: SEBICategory;
  title?: string;
  date?: string;
  url?: string;
}

interface IngestionOverrides {
  sourcePath?: string;
}

function createIngestion(overrides: IngestionOverrides = {}): CorpusIngestion {
  const openAiKey = getEnvVar('OPENAI_API_KEY');
  const qdrantUrl = getEnvVar('QDRANT_URL');
  const qdrantApiKey = process.env.QDRANT_API_KEY;
  const sourcePath = overrides.sourcePath ?? process.env.CORPUS_SOURCE_PATH;

  const embedder = new EmbeddingGenerator(openAiKey, {
    logger: console,
  });
  const qdrant = new QdrantManager({
    url: qdrantUrl,
    apiKey: qdrantApiKey,
    logger: console,
  });

  return new CorpusIngestion(
    {
      embedder,
      qdrant,
    },
    {
      sourcePath,
      logger: console,
    },
  );
}

function buildMetadataOverrides(options: FileOptions): Partial<SEBIDocument> | undefined {
  const overrides: Partial<SEBIDocument> = {};

  if (options.circularId) {
    overrides.circular_id = options.circularId;
  }

  if (options.category) {
    if (!CATEGORY_OPTIONS.includes(options.category)) {
      throw new Error(`Invalid category: ${options.category}`);
    }
    overrides.category = options.category;
  }

  if (options.title) {
    overrides.title = options.title;
  }

  if (options.date) {
    const parsed = new Date(options.date);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error(`Invalid date: ${options.date}`);
    }
    overrides.date = parsed;
  }

  if (options.url) {
    overrides.url = options.url;
  }

  return Object.keys(overrides).length > 0 ? overrides : undefined;
}

function getEnvVar(key: string): string {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}

async function runWithSpinner<T>(text: string, task: () => Promise<T>): Promise<T> {
  const spinner = ora(text).start();
  try {
    const result = await task();
    spinner.succeed(text);
    return result;
  } catch (error) {
    spinner.fail(text);
    throw error;
  }
}

function printSummary(summary: IngestionSummary): void {
  console.log(chalk.green(`✔ ${summary.circularId} ingested`));
  console.log(`  chunks: ${summary.chunkCount}`);
  console.log(`  embeddings: ${summary.embeddingCount}`);
  console.log(`  duration: ${(summary.durationMs / 1000).toFixed(2)}s`);
}

program.parseAsync(process.argv).catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(chalk.red(message));
  process.exitCode = 1;
});
