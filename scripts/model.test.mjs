import assert from "node:assert/strict";
import { resolve } from "node:path";
import { test } from "node:test";

import { env, pipeline } from "@huggingface/transformers";

import manifest from "../models/qwen3-embedding-0.6b/manifest.json" with { type: "json" };

test("the offline CPU pipeline produces normalized embeddings for retrieval", async () => {
  assert.equal(env.version, manifest.runtime.version);
  assert.equal(manifest.runtime.task, "feature-extraction");
  assert.equal(manifest.runtime.device, "cpu");
  assert.equal(manifest.runtime.dtype, "q8");
  assert.equal(manifest.runtime.recommendedBatchSize, 1);
  assert.equal(manifest.model.pooling, "last_token");
  env.allowRemoteModels = false;
  env.useFSCache = false;

  const directory = resolve(
    process.env.MODEL_DIR ?? `.artifacts/${manifest.modelId}`,
  );
  const extractor = await pipeline("feature-extraction", directory, {
    local_files_only: true,
    device: "cpu",
    dtype: "q8",
    subfolder: manifest.runtime.subfolder,
    session_options: { intraOpNumThreads: 2, interOpNumThreads: 1 },
  });

  try {
    const texts = [
      "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
      "The capital of China is Beijing.",
      "Gravity is responsible for the movement of planets around the sun.",
    ];
    /** @param {Float32Array} left @param {Float32Array} right */
    const dot = (left, right) =>
      left.reduce((sum, value, index) => sum + value * right[index], 0);
    const vectors = [];
    for (const text of texts) {
      const embedding = await extractor(text, {
        pooling: "last_token",
        normalize: manifest.model.normalization === "l2",
      });
      assert.deepEqual(embedding.dims, [1, manifest.model.embeddingDimensions]);
      const { data } = embedding;
      assert.ok(data instanceof Float32Array);
      assert.ok(data.every(Number.isFinite));
      assert.ok(
        Math.abs(dot(data, data) - 1) < 1e-4,
        "embedding must have unit L2 norm",
      );
      vectors.push(data);
    }
    const [query, relevant, unrelated] = vectors;
    assert.ok(dot(query, relevant) > dot(query, unrelated));
  } finally {
    await extractor.dispose();
  }
});
