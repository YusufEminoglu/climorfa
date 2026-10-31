#!/usr/bin/env node
/**
 * js_reference.mjs
 * Reference runner for 02Geo3D wind.js under Node.js (v26+).
 *
 * Runs the ORIGINAL plugin/wind.js on synthetic grids and real tiles,
 * reading binary float32 heights/porosity arrays and saving raw float32 ratio arrays
 * for non-circular verification against the Python/Numba port.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import ORIGINAL wind.js directly from plugin/ via pathToFileURL for Windows ESM support
const windJsPath = path.resolve(__dirname, '../../../plugin/wind.js');
const { windRatio, OPEN_PEDESTRIAN_FACTOR, lawsonClass } = await import(pathToFileURL(windJsPath).href);

async function runTask(task) {
  const { id, heights_path, porosity_path, cols, rows, cell, windFromDeg, output_path } = task;

  const heightsBuf = fs.readFileSync(heights_path);
  const heights = new Float32Array(heightsBuf.buffer, heightsBuf.byteOffset, heightsBuf.byteLength / 4);

  let porosity = null;
  if (porosity_path && fs.existsSync(porosity_path)) {
    const poroBuf = fs.readFileSync(porosity_path);
    porosity = new Float32Array(poroBuf.buffer, poroBuf.byteOffset, poroBuf.byteLength / 4);
  }

  const res = await windRatio({
    heights,
    porosity,
    cols,
    rows,
    cell,
    windFromDeg,
  });

  const outBuf = Buffer.from(res.buffer, res.byteOffset, res.byteLength);
  fs.mkdirSync(path.dirname(output_path), { recursive: true });
  fs.writeFileSync(output_path, outBuf);
}

async function main() {
  const args = process.argv.slice(2);
  let manifestPath = null;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--manifest' && i + 1 < args.length) {
      manifestPath = args[i + 1];
    }
  }

  if (!manifestPath) {
    console.error('Usage: node js_reference.mjs --manifest <manifest.json>');
    process.exit(1);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const total = manifest.tasks.length;
  console.log(`[js_reference] Running ${total} tasks using ORIGINAL plugin/wind.js...`);

  const t0 = performance.now();
  for (let i = 0; i < total; i++) {
    const task = manifest.tasks[i];
    await runTask(task);
  }
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`[js_reference] Completed ${total} tasks in ${elapsed}s.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
