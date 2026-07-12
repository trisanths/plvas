import assert from "node:assert/strict";
import test from "node:test";

import {
  computeOcrDetectorSize,
  decodeCtc,
  extractOcrBoxes,
  parseOcrDictionary,
} from "../src/ocr/postprocess.js";
import {
  associateContextualIdentityRegions,
  associateCueValues,
} from "../src/ocr/association.js";
import {
  detectSensitiveCues,
  detectSensitiveValueCue,
} from "../src/ocr/rules.js";
import {
  applyScreenshotRedactionPolicy,
  filterContextualHits,
} from "../src/ocr/policy.js";
import { runOrtSerialized, selectWasmThreadCount } from "../src/runtime.js";
import { fuseSensitiveRegions } from "../src/visual/fusion.js";

test("caps an ultra-wide OCR detector tensor and rounds to multiples of 32", () => {
  assert.deepEqual(computeOcrDetectorSize(1600, 280), {
    width: 2048,
    height: 352,
    scaleX: 1.28,
    scaleY: 352 / 280,
  });
});

test("uses bounded WASM threading only in a cross-origin-isolated runtime", () => {
  assert.equal(
    selectWasmThreadCount({ isolated: false, requested: 8, hardwareConcurrency: 16 }),
    1,
  );
  assert.equal(
    selectWasmThreadCount({ isolated: true, requested: 8, hardwareConcurrency: 6 }),
    6,
  );
  assert.equal(
    selectWasmThreadCount({ isolated: true, requested: 99, hardwareConcurrency: 32 }),
    8,
  );
});

test("serializes shared WASM runtime inference without poisoning the queue", async () => {
  const events = [];
  const first = runOrtSerialized(async () => {
    events.push("first-start");
    await Promise.resolve();
    events.push("first-end");
    throw new Error("expected test failure");
  });
  const second = runOrtSerialized(async () => {
    events.push("second-start");
    await Promise.resolve();
    events.push("second-end");
    return 2;
  });
  await assert.rejects(first, /expected test failure/);
  assert.equal(await second, 2);
  assert.deepEqual(events, ["first-start", "first-end", "second-start", "second-end"]);
});

test("keeps the dictionary's literal space and appends RapidOCR's second space", () => {
  assert.deepEqual(parseOcrDictionary("a\n \n"), ["<blank>", "a", " ", " "]);
});

test("CTC collapses duplicate classes but preserves repeats separated by blank", () => {
  const dictionary = ["<blank>", "a", "b", " "];
  const sequence = [1, 1, 0, 1, 2, 0, 3];
  const values = new Float32Array(sequence.length * dictionary.length);
  for (let step = 0; step < sequence.length; step += 1) {
    values[step * dictionary.length + sequence[step]] = 0.9;
  }
  const decoded = decodeCtc(
    values,
    [1, sequence.length, dictionary.length],
    dictionary,
  );
  assert.equal(decoded.text, "aab ");
  assert.ok(Math.abs(decoded.confidence - 0.9) < 1e-6);
});

test("extracts and conservatively expands a connected OCR text component", () => {
  const width = 12;
  const height = 8;
  const probability = new Float32Array(width * height);
  for (let y = 2; y <= 4; y += 1) {
    for (let x = 2; x <= 7; x += 1) probability[y * width + x] = 0.9;
  }
  const boxes = extractOcrBoxes(probability, width, height, width, height);
  assert.equal(boxes.length, 1);
  assert.ok(boxes[0].x1 < 2);
  assert.ok(boxes[0].y1 < 2);
  assert.ok(boxes[0].x2 > 8);
  assert.ok(boxes[0].y2 > 5);
});

test("secret cue rules catch OCR-tolerant CVC and credential labels", () => {
  assert.deepEqual(
    detectSensitiveCues("Card number 4000 0000 0000 0002 Security c0de 007"),
    ["CVC", "CARD_NUMBER"],
  );
  assert.deepEqual(detectSensitiveCues("API key sk_test_123"), ["API_KEY"]);
  assert.deepEqual(detectSensitiveCues("Order 4242-2026 total $412.30"), []);
  assert.deepEqual(
    detectSensitiveCues("Recovery phone: +1 (202) 555-0812 ext 700012"),
    ["PHONE"],
  );
  assert.deepEqual(
    detectSensitiveCues("eyJhbGciOiJub25lIn0.eyJzdWIiOiJmYWtlIn0.signature"),
    ["AUTH_TOKEN"],
  );
});

test("recognizes cue-only form labels without treating ordinary copy as a cue", () => {
  assert.equal(detectSensitiveValueCue("Email address *"), "EMAIL");
  assert.equal(detectSensitiveValueCue("Date of birth:"), "DOB");
  assert.equal(detectSensitiveValueCue("Company name"), null);
  assert.equal(detectSensitiveValueCue("How did you hear about us?"), null);
});

test("associates separate form labels with the nearest right-hand or lower value", () => {
  const associated = associateCueValues([
    { x1: 20, y1: 20, x2: 110, y2: 44, text: "Email", ocrConfidence: 0.99 },
    { x1: 140, y1: 20, x2: 360, y2: 44, text: "alex@example.com", ocrConfidence: 0.93 },
    { x1: 20, y1: 100, x2: 160, y2: 124, text: "Date of birth", ocrConfidence: 0.98 },
    { x1: 20, y1: 140, x2: 180, y2: 164, text: "01/02/1999", ocrConfidence: 0.91 },
  ]);

  assert.deepEqual(
    associated.map(({ label, x1, y1, x2, y2, sources }) => ({ label, x1, y1, x2, y2, sources })),
    [
      { label: "EMAIL", x1: 140, y1: 20, x2: 360, y2: 44, sources: ["OCR+CUE_ASSOCIATION"] },
      { label: "DOB", x1: 20, y1: 140, x2: 180, y2: 164, sources: ["OCR+CUE_ASSOCIATION"] },
    ],
  );
});

test("does not associate a distant value or reuse one value for two cues", () => {
  const associated = associateCueValues([
    { x1: 10, y1: 10, x2: 80, y2: 30, text: "Email" },
    { x1: 10, y1: 35, x2: 80, y2: 55, text: "Phone" },
    { x1: 100, y1: 20, x2: 260, y2: 42, text: "one@example.com", ocrConfidence: 0.9 },
    { x1: 900, y1: 900, x2: 980, y2: 920, text: "too far", ocrConfidence: 0.9 },
  ]);
  assert.equal(associated.length, 1);
});

test("rejects nearby UI controls and implausible cue values", () => {
  const associated = associateCueValues([
    { x1: 10, y1: 10, x2: 90, y2: 30, text: "Email" },
    { x1: 110, y1: 10, x2: 210, y2: 30, text: "Continue", ocrConfidence: 0.99 },
    { x1: 10, y1: 60, x2: 90, y2: 80, text: "CVC" },
    { x1: 110, y1: 60, x2: 210, y2: 80, text: "Order total", ocrConfidence: 0.99 },
    { x1: 10, y1: 110, x2: 90, y2: 130, text: "Account" },
    { x1: 110, y1: 110, x2: 210, y2: 130, text: "Settings", ocrConfidence: 0.99 },
  ]);
  assert.deepEqual(associated, []);
});

test("does not guess secret values from a nearby unrelated OCR box", () => {
  const associated = associateCueValues([
    { x1: 10, y1: 10, x2: 100, y2: 30, text: "Password" },
    { x1: 120, y1: 10, x2: 220, y2: 30, text: "Sign in", ocrConfidence: 0.99 },
  ]);
  assert.deepEqual(associated, []);
});

test("projects tightly scoped commerce identity headers into redaction regions", () => {
  const contextual = associateContextualIdentityRegions([
    { x1: 928, y1: 7, x2: 1023, y2: 35, text: "Hello,Angela", ocrConfidence: 0.8 },
    { x1: 163, y1: 9, x2: 284, y2: 38, text: "Deliver to Angela", ocrConfidence: 0.9 },
    { x1: 10, y1: 106, x2: 763, y2: 136, text: "Angela pay for this order Get a gift card", ocrConfidence: 0.93 },
  ]);

  assert.deepEqual(
    contextual.map(({ label }) => label),
    ["NAME", "NAME", "ADDRESS", "NAME"],
  );
  assert.ok(covers(contextual, [972.3, 12.5, 1013.9, 26.5]));
  assert.ok(covers(contextual, [232.4, 15.5, 273.9, 29.5]));
  assert.ok(covers(contextual, [170, 28.5, 251.3, 44.5]));
  assert.ok(covers(contextual, [256.1, 28.5, 304.9, 44.5]));
  assert.ok(covers(contextual, [15, 111, 69.6, 127]));
  assert.ok(contextual.every(({ sources }) => sources[0] === "OCR+CONTEXT_RULE"));
});

test("commerce identity rules cover multi-token names", () => {
  const contextual = associateContextualIdentityRegions([
    { x1: 100, y1: 10, x2: 340, y2: 34, text: "Hello, Angela Smith", ocrConfidence: 0.9 },
    { x1: 10, y1: 60, x2: 410, y2: 84, text: "Angela Smith pay for this order", ocrConfidence: 0.9 },
    { x1: 100, y1: 110, x2: 460, y2: 134, text: "Hello, Angela de la Cruz", ocrConfidence: 0.9 },
    { x1: 10, y1: 160, x2: 520, y2: 184, text: "Ludwig van der Meer pay for this order", ocrConfidence: 0.9 },
    { x1: 100, y1: 210, x2: 520, y2: 234, text: "Hi Mary Jane Alice Smith", ocrConfidence: 0.9 },
  ]);
  assert.equal(contextual.length, 5);
  assert.ok(covers(contextual, [190, 12, 330, 32]));
  assert.ok(covers(contextual, [12, 62, 175, 82]));
  assert.ok(covers(contextual, [220, 112, 450, 132]));
  assert.ok(covers(contextual, [12, 162, 270, 182]));
  assert.ok(covers(contextual, [140, 212, 510, 232]));
});

test("commerce identity rules reject generic greeting and delivery copy", () => {
  const generic = [
    "Hello world",
    "Hello, there",
    "Deliver to your door",
    "Deliver to this address",
    "Deliver to Store",
    "Deliver to Locker",
    "Deliver to Pickup",
    "Deliver to Office",
    "Hello, Support",
    "Welcome back Admin",
    "We deliver to 50 states",
    "Company pay for this order",
    "Angela pay for this invoice",
  ].map((text, index) => ({
    x1: 0,
    y1: index * 20,
    x2: 200,
    y2: index * 20 + 18,
    text,
    ocrConfidence: 0.99,
  }));
  assert.deepEqual(associateContextualIdentityRegions(generic), []);
});

test("screenshot policy redacts city, state, and postal code", () => {
  const spans = [
    { start: 0, end: 7, label: "CITY", score: 0.9 },
    { start: 8, end: 10, label: "STATE", score: 0.9 },
    { start: 11, end: 16, label: "ZIP_CODE", score: 0.9 },
  ];
  assert.deepEqual(
    new Set(applyScreenshotRedactionPolicy(spans).map(({ label }) => label)),
    new Set(["CITY", "STATE", "ZIP_CODE"]),
  );
});

test("drops an isolated address-component guess from developer status text", () => {
  const hit = { label: "BUILDING_NUMBER", start: 52, end: 57 };
  assert.deepEqual(
    filterContextualHits(
      [hit],
      "localhost:5173 commit a1b2c3d4 version 2026.07 PID 55301",
    ),
    [],
  );
  assert.deepEqual(
    filterContextualHits(
      [hit, { label: "STREET_NAME", start: 4, end: 15 }],
      "275 Sara Summit",
    ).length,
    2,
  );
});

test("fuses a visual box contained by an OCR-semantic line and preserves evidence", () => {
  const fused = fuseSensitiveRegions(
    [{ x1: 100, y1: 20, x2: 220, y2: 50, label: "CARD_NUMBER", score: 0.8 }],
    [{
      x1: 40,
      y1: 15,
      x2: 300,
      y2: 55,
      label: "CVC + CREDIT_CARD",
      labels: ["CVC", "CREDIT_CARD"],
      sources: ["OCR+RAMPART", "OCR+RULE"],
      score: 0.9,
    }],
  );
  assert.equal(fused.length, 1);
  assert.deepEqual(
    new Set(fused[0].sources),
    new Set(["OCR+RAMPART", "OCR+RULE", "VISUAL"]),
  );
  assert.deepEqual(
    new Set(fused[0].labels),
    new Set(["CVC", "CREDIT_CARD", "CARD_NUMBER"]),
  );
  assert.deepEqual(
    [fused[0].x1, fused[0].y1, fused[0].x2, fused[0].y2],
    [40, 15, 300, 55],
  );
});

function covers(regions, truth) {
  return regions.some((region) => {
    const intersection =
      Math.max(0, Math.min(region.x2, truth[2]) - Math.max(region.x1, truth[0])) *
      Math.max(0, Math.min(region.y2, truth[3]) - Math.max(region.y1, truth[1]));
    const area = (truth[2] - truth[0]) * (truth[3] - truth[1]);
    return intersection / area >= 0.98;
  });
}
