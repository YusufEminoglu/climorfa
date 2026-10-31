/**
 * WP0.3 Step 2: Run 02Geo3D comfort.js calculations via Node.js ESM.
 *
 * Source: C:\Users\YE\PyCharmMiscProject\qgis_plugins\zero2geospatial3d\web\src\core\comfort.js
 *
 * Plugin Default Parameters for PET:
 * - met: 80 W (metabolic activity heat in Watts, added to male basal metabolic rate metab_male ~86.8 W)
 * - clo: 0.9 (clothing insulation)
 * - Internally fixed in comfort.js:
 *   - age: 35 years
 *   - weight: 75.0 kg
 *   - height: 1.8 m
 *   - posture: sitting (feff = 0.725, hc = 2.67 + 6.5 * v^0.67)
 *   - p_atm: 1013.25 hPa
 *
 * Plugin Parameters for UTCI:
 * - 6th-order polynomial approximation of Broede et al. (2012)
 * - limit: false by default (evaluates polynomial directly); limit: true returns NaN outside validity bounds
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const PLUGIN_SRC = 'C:/Users/YE/PyCharmMiscProject/qgis_plugins/zero2geospatial3d/web/src/core/comfort.js';
const INPUT_CSV = path.resolve('outputs/street/crosscheck/comfort_inputs_SYNTHETIC.csv');
const OUTPUT_CSV = path.resolve('outputs/street/crosscheck/comfort_plugin_predictions_SYNTHETIC.csv');

async function main() {
  console.log(`Loading comfort.js from ${PLUGIN_SRC}...`);
  const pluginUrl = pathToFileURL(PLUGIN_SRC).href;
  const comfort = await import(pluginUrl);

  console.log(`Reading input samples from ${INPUT_CSV}...`);
  const rawData = fs.readFileSync(INPUT_CSV, 'utf-8');
  const lines = rawData.trim().split(/\r?\n/);
  const header = lines[0].split(',');

  const idxSampleId = header.indexOf('sample_id');
  const idxSampleSet = header.indexOf('sample_set');
  const idxSynthetic = header.indexOf('is_synthetic');
  const idxTdb = header.indexOf('tdb');
  const idxDeltaTmrt = header.indexOf('delta_tmrt');
  const idxTr = header.indexOf('tr');
  const idxV = header.indexOf('v');
  const idxRh = header.indexOf('rh');
  const idxIzmir = header.indexOf('in_izmir_summer_domain');

  console.log(`Processing ${lines.length - 1} rows with comfort.js...`);
  const startTime = Date.now();

  const outHeader = [
    ...header,
    'utci_plugin',
    'utci_plugin_limit',
    'utci_category_plugin',
    'pet_plugin',
    'pet_category_plugin'
  ];

  const outLines = [outHeader.join(',')];

  let nanUtciCount = 0;
  let nanPetCount = 0;

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    const sampleId = cols[idxSampleId];
    const sampleSet = cols[idxSampleSet];
    const isSynthetic = cols[idxSynthetic];
    const tdb = parseFloat(cols[idxTdb]);
    const deltaTmrt = parseFloat(cols[idxDeltaTmrt]);
    const tr = parseFloat(cols[idxTr]);
    const v = parseFloat(cols[idxV]);
    const rh = parseFloat(cols[idxRh]);
    const inIzmir = cols[idxIzmir];

    // Compute UTCI (unlimited and limited)
    const utciVal = comfort.utci(tdb, tr, v, rh, { limit: false });
    const utciLimitVal = comfort.utci(tdb, tr, v, rh, { limit: true });
    const utciCat = comfort.utciCategory(utciVal)?.label ?? '';

    // Compute PET (default met=80, clo=0.9)
    const petVal = comfort.pet(tdb, tr, v, rh, { met: 80, clo: 0.9 });
    const petCat = comfort.petCategory(petVal)?.label ?? '';

    if (Number.isNaN(utciVal)) nanUtciCount++;
    if (Number.isNaN(petVal)) nanPetCount++;

    const outRow = [
      sampleId,
      sampleSet,
      isSynthetic,
      tdb,
      deltaTmrt,
      tr,
      v,
      rh,
      inIzmir,
      Number.isFinite(utciVal) ? utciVal.toFixed(6) : 'NaN',
      Number.isFinite(utciLimitVal) ? utciLimitVal.toFixed(6) : 'NaN',
      `"${utciCat}"`,
      Number.isFinite(petVal) ? petVal.toFixed(6) : 'NaN',
      `"${petCat}"`
    ];

    outLines.push(outRow.join(','));
  }

  const durationMs = Date.now() - startTime;
  console.log(`Calculations finished in ${durationMs} ms (${(durationMs / (lines.length - 1)).toFixed(3)} ms/row).`);
  console.log(`NaN counts: UTCI=${nanUtciCount}, PET=${nanPetCount}`);

  fs.writeFileSync(OUTPUT_CSV, outLines.join('\n'), 'utf-8');
  console.log(`Successfully saved predictions to ${OUTPUT_CSV}`);
}

main().catch((err) => {
  console.error('Fatal error running plugin calculations:', err);
  process.exit(1);
});
