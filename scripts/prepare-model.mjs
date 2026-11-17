import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { copyFile, mkdir, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { setTimeout as delay } from "node:timers/promises";

import manifest from "../models/qwen3-embedding-0.6b/manifest.json" with { type: "json" };

const manifestFile = new URL(
  "../models/qwen3-embedding-0.6b/manifest.json",
  import.meta.url,
);
const repoRoot = new URL("../", import.meta.url);
const upstream = new URL(
  `https://huggingface.co/${manifest.upstream.repository}/resolve/${manifest.upstream.revision}/`,
);

/** @param {string} file */
async function sha256(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

/** @param {URL} url @param {string} destination */
async function download(url, destination) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(300_000),
      });
      if (!response.ok || !response.body) {
        await response.body?.cancel();
        throw new Error(`Download failed for ${url}: HTTP ${response.status}`);
      }
      await pipeline(
        Readable.fromWeb(response.body),
        createWriteStream(destination),
      );
      return;
    } catch (error) {
      if (attempt === 3) throw error;
      console.error(`Retrying ${url.pathname}: ${error}`);
      await delay(attempt * 1_000);
    }
  }
}

const output = resolve(process.argv[2] ?? `.artifacts/${manifest.modelId}`);
await mkdir(dirname(output), { recursive: true });
await mkdir(output);

try {
  const checksums = [];
  for (const artifact of manifest.artifacts) {
    const destination = join(output, artifact.name);
    if (artifact.upstreamPath) {
      await download(new URL(artifact.upstreamPath, upstream), destination);
    } else if (artifact.repositoryPath) {
      await copyFile(new URL(artifact.repositoryPath, repoRoot), destination);
    } else {
      throw new Error(`Missing source for ${artifact.name}`);
    }

    const { size } = await stat(destination);
    const digest = await sha256(destination);
    if (size !== artifact.sizeBytes || digest !== artifact.sha256) {
      throw new Error(
        `Integrity check failed for ${artifact.name}: ${size} bytes, SHA-256 ${digest}`,
      );
    }
    checksums.push(`${digest}  ${artifact.name}\n`);
    console.log(`Verified ${artifact.name} (${size} bytes)`);
  }

  const manifestName = "manifest.json";
  const preparedManifest = join(output, manifestName);
  await copyFile(manifestFile, preparedManifest);
  checksums.push(`${await sha256(preparedManifest)}  ${manifestName}\n`);
  await writeFile(join(output, manifest.release.checksums), checksums.join(""));
  console.log(`Prepared ${manifest.release.tag} at ${output}`);
} catch (error) {
  await rm(output, { recursive: true, force: true });
  throw error;
}
