# NekoSearch models

Official model downloads for [NekoSearch](https://github.com/rafaself/nekosearch).
Each version is published as an immutable GitHub Release with a manifest, SHA-256 checksums, and license and attribution files.

| Model                | Release                                                                                          | Runtime                               |
| -------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------- |
| Qwen3-Embedding-0.6B | [v1.0.0](https://github.com/rafaself/nekosearch-models/releases/tag/qwen3-embedding-0.6b-v1.0.0) | Transformers.js `q8`, Node.js 24, CPU |

## Download and verify

Download the assets into a fresh directory. These URLs are public and require no GitHub token.

```bash
set -e
mkdir qwen3-embedding-0.6b
cd qwen3-embedding-0.6b
base=https://github.com/rafaself/nekosearch-models/releases/download/qwen3-embedding-0.6b-v1.0.0

for file in config.json tokenizer.json tokenizer_config.json model_quantized.onnx \
  Apache-2.0.txt NOTICE manifest.json SHA256SUMS; do
  curl --disable --fail --location --retry 3 --output "$file" "$base/$file"
done

sha256sum --check --strict SHA256SUMS
```

The [tagged source manifest](https://github.com/rafaself/nekosearch-models/blob/qwen3-embedding-0.6b-v1.0.0/models/qwen3-embedding-0.6b/manifest.json)
provides the expected byte counts and model asset hashes separately from the release downloads.
`SHA256SUMS` also covers the downloaded manifest.

## Load with Transformers.js

The four inference files are `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model_quantized.onnx`.
Keep them in the same directory and set `subfolder: ""` to use the flat release layout.

Tested with Node.js 24.22.2, `@huggingface/transformers@4.3.2`, and ONNX Runtime Node 1.30.0 on Linux x64 CPU.

```js
import { resolve } from "node:path";
import { env, pipeline } from "@huggingface/transformers";

env.allowRemoteModels = false;
const embed = await pipeline(
  "feature-extraction",
  resolve("./qwen3-embedding-0.6b"),
  {
    device: "cpu",
    dtype: "q8",
    subfolder: "",
    local_files_only: true,
  },
);

try {
  const vector = await embed("NekoSearch indexes framework documentation.", {
    pooling: "last_token",
    normalize: true,
  });
  console.log(vector.dims); // [1, 1024]
} finally {
  await embed.dispose();
}
```

Embed one input at a time for consistent vectors across indexing and querying. Mixed-length batches changed vectors in the CPU compatibility check. Keep inputs within the model's 32,768-token context. Use the original model's [instruction format](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md) for queries, and pass documents as plain text.

## Reproduce and publish

```bash
pnpm install --frozen-lockfile
node scripts/prepare-model.mjs
pnpm test
```

Run from the repository root with Node.js and pnpm versions pinned in `.nvmrc` and `package.json`. The preparation script downloads the four pinned upstream files, checks their size and SHA-256, and writes the release files to a new directory under `.artifacts/`. Pass a path as the first argument to use a different new output directory. Model weights are stored exclusively in GitHub Release assets for distribution.

For a new model version, update the manifest and versioned download links, then push the matching tag after validation. Keep release immutability enabled in the repository settings. GitHub Actions tests the model with remote loading disabled, rebuilds the files from the pinned sources, verifies GitHub's uploaded asset digests, and checks the immutable flag after publication. Run the workflow manually from the default branch with an existing tag to retry publication.

## Provenance and license

The [original Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md) declares Apache-2.0. The [pinned ONNX conversion](https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/tree/c25a394dd583836952667c12f008335071b3f43d) has no separate license declaration. [Xenova, who uploaded this conversion](https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/discussions/1), [states that his quantizations follow the original model license](https://huggingface.co/Xenova/bge-m3/discussions/9). The model release ships the [Apache license text](licenses/Apache-2.0.txt) and [NOTICE](NOTICE). The repository's [MIT license](LICENSE) covers the NekoSearch distribution tooling and documentation.
