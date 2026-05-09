#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const runtimeRoot = path.join(
  process.env.HOME,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs",
);

const artifact = await import(pathToFileURL(runtimeRoot).href);
const { Presentation, PresentationFile } = artifact;

const W = 1280;
const H = 720;
const OUT = path.join(__dirname, "Customer_Churn_Prediction_with_MLflow.pptx");
const SCRIPT_OUT = path.join(__dirname, "presentation_script.md");
const PREVIEW_DIR = path.join(__dirname, "preview");

const colors = {
  bg: "#08111F",
  panel: "#101C2D",
  panel2: "#15263A",
  text: "#F8FAFC",
  muted: "#A8B3C7",
  teal: "#43D3A6",
  yellow: "#F4C95D",
  blue: "#72A7FF",
  red: "#F97373",
  line: "#284057",
  white: "#FFFFFF",
};

const fonts = {
  title: "Aptos Display",
  body: "Aptos",
  mono: "Aptos Mono",
};

const imagePath = (name) => path.join(projectRoot, "screenshots", name);

async function readImageBlob(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, { x, y, w, h, fill = colors.panel, line = colors.line, width = 1, name }) {
  const shape = slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width },
  });
  return shape;
}

function addText(slide, text, { x, y, w, h, size = 24, color = colors.text, bold = false, face = fonts.body, align = "left", valign = "top", fill = "#00000000", line = "#00000000", inset = 0 }) {
  const shape = addShape(slide, { x, y, w, h, fill, line, width: 0 });
  shape.text = text;
  shape.text.fontSize = size;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.typeface = face;
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = { left: inset, right: inset, top: inset, bottom: inset };
  return shape;
}

async function addImage(slide, filePath, { x, y, w, h, fit = "contain", alt = "" }) {
  const image = slide.images.add({
    blob: await readImageBlob(filePath),
    fit,
    alt,
  });
  image.position = { left: x, top: y, width: w, height: h };
  return image;
}

function addKicker(slide, text) {
  addText(slide, text.toUpperCase(), {
    x: 64,
    y: 42,
    w: 860,
    h: 26,
    size: 15,
    color: colors.teal,
    bold: true,
  });
}

function addFooter(slide, n) {
  addText(slide, "AIN-3009 MLOps Term Project", {
    x: 64,
    y: 676,
    w: 360,
    h: 22,
    size: 13,
    color: colors.muted,
  });
  addText(slide, String(n).padStart(2, "0"), {
    x: 1160,
    y: 672,
    w: 56,
    h: 28,
    size: 14,
    color: colors.teal,
    bold: true,
    align: "right",
  });
}

function addTitle(slide, title, subtitle) {
  addKicker(slide, subtitle);
  addText(slide, title, {
    x: 64,
    y: 92,
    w: 700,
    h: 132,
    size: 48,
    color: colors.text,
    bold: true,
    face: fonts.title,
  });
}

function addBulletList(slide, items, { x, y, w, h, size = 24 }) {
  const text = items.map((item) => `- ${item}`).join("\n");
  addText(slide, text, {
    x,
    y,
    w,
    h,
    size,
    color: colors.muted,
    inset: 4,
  });
}

function addMetricCard(slide, value, label, { x, y, w = 220, h = 122, accent = colors.teal }) {
  addShape(slide, { x, y, w, h, fill: colors.panel, line: colors.line, width: 1 });
  addShape(slide, { x, y, w: 5, h, fill: accent, line: accent, width: 0 });
  addText(slide, value, {
    x: x + 22,
    y: y + 20,
    w: w - 42,
    h: 42,
    size: 32,
    color: colors.text,
    bold: true,
    face: fonts.title,
  });
  addText(slide, label, {
    x: x + 22,
    y: y + 68,
    w: w - 42,
    h: 42,
    size: 15,
    color: colors.muted,
  });
}

function addScreenshotFrame(slide, caption, { x, y, w, h }) {
  addShape(slide, { x, y, w, h, fill: "#07101C", line: colors.line, width: 1 });
  addText(slide, caption, {
    x: x + 16,
    y: y + h - 34,
    w: w - 32,
    h: 22,
    size: 13,
    color: colors.muted,
  });
}

function setNotes(slide, note) {
  slide.speakerNotes.setText(note);
}

function background(slide) {
  slide.background.fill = colors.bg;
  addShape(slide, { x: 0, y: 0, w: W, h: H, fill: colors.bg, line: colors.bg, width: 0 });
  addShape(slide, { x: 978, y: 0, w: 302, h: H, fill: "#0D1B2C", line: "#0D1B2C", width: 0 });
  addShape(slide, { x: 64, y: 648, w: 1088, h: 1, fill: colors.line, line: colors.line, width: 0 });
}

function newSlide(presentation, n) {
  const slide = presentation.slides.add();
  background(slide);
  addFooter(slide, n);
  return slide;
}

async function build() {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const script = ["# Presentation Script", ""];

  let slide = newSlide(presentation, 1);
  addText(slide, "AIN-3009 Delivering AI Applications with MLOps", { x: 64, y: 54, w: 740, h: 26, size: 16, color: colors.teal, bold: true });
  addText(slide, "Customer Churn Prediction with MLflow", { x: 64, y: 116, w: 790, h: 134, size: 54, color: colors.text, bold: true, face: fonts.title });
  addText(slide, "End-to-end machine learning lifecycle management for telecom churn: tracking, tuning, registry, serving, and monitoring.", { x: 66, y: 278, w: 720, h: 80, size: 23, color: colors.muted });
  addMetricCard(slide, "7,043", "customer records", { x: 64, y: 430, accent: colors.teal });
  addMetricCard(slide, "63", "MLflow runs logged", { x: 316, y: 430, accent: colors.blue });
  addMetricCard(slide, "v1", "production model", { x: 568, y: 430, accent: colors.yellow });
  addText(slide, "Ayhan Gurbangeldiyev\nTerm Project", { x: 978, y: 520, w: 230, h: 62, size: 18, color: colors.text, bold: true, align: "right" });
  setNotes(slide, "Hello, today I will present my MLOps term project: Customer Churn Prediction with MLflow. The goal is not only to train a model, but to manage the full machine learning lifecycle from experiments to deployment and monitoring.");
  script.push("## Slide 1 - Title", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 2);
  addTitle(slide, "Predict customers likely to leave", "Problem and dataset");
  addBulletList(slide, [
    "Domain: telecommunications customer retention.",
    "Dataset: Telco Customer Churn with 7,043 rows and 21 columns.",
    "Target: Churn, encoded as a binary Yes or No outcome.",
    "Business value: identify high-risk customers before cancellation.",
  ], { x: 72, y: 260, w: 650, h: 230, size: 24 });
  addMetricCard(slide, "5,174", "non-churn customers", { x: 820, y: 120, w: 260, accent: colors.blue });
  addMetricCard(slide, "1,869", "churn customers", { x: 820, y: 270, w: 260, accent: colors.red });
  addMetricCard(slide, "26.5%", "positive class share", { x: 820, y: 420, w: 260, accent: colors.teal });
  setNotes(slide, "The selected problem is customer churn prediction in telecommunications. The model predicts whether a customer will leave. This is useful because companies can use churn risk to prioritize retention campaigns.");
  script.push("## Slide 2 - Problem and Dataset", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 3);
  addTitle(slide, "One repeatable MLflow pipeline", "End-to-end workflow");
  const steps = [
    ["1", "Preprocess", "Clean TotalCharges, drop customerID, encode Churn"],
    ["2", "Train", "Compare Logistic Regression, Random Forest, Gradient Boosting"],
    ["3", "Tune", "RandomizedSearchCV with 20 candidates and 3-fold CV"],
    ["4", "Register", "Promote ChurnPredictionModel to staging and production"],
    ["5", "Serve", "Expose production model through MLflow Models"],
    ["6", "Monitor", "Log batch-level drift and performance metrics"],
  ];
  steps.forEach(([num, title, body], idx) => {
    const col = idx % 3;
    const row = Math.floor(idx / 3);
    const x = 76 + col * 360;
    const y = 260 + row * 148;
    addShape(slide, { x, y, w: 310, h: 112, fill: colors.panel, line: colors.line });
    addText(slide, num, { x: x + 18, y: y + 18, w: 42, h: 42, size: 24, color: colors.teal, bold: true, align: "center", valign: "middle", fill: "#102D34" });
    addText(slide, title, { x: x + 74, y: y + 17, w: 210, h: 28, size: 22, color: colors.text, bold: true });
    addText(slide, body, { x: x + 74, y: y + 52, w: 210, h: 42, size: 14, color: colors.muted });
  });
  setNotes(slide, "This is the full lifecycle workflow. The project starts with preprocessing, then baseline training, hyperparameter tuning, model registry, serving, and finally simulated monitoring.");
  script.push("## Slide 3 - Workflow", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 4);
  addTitle(slide, "Preprocessing keeps training and inference consistent", "Data preparation and models");
  addShape(slide, { x: 72, y: 260, w: 494, h: 258, fill: colors.panel, line: colors.line });
  addText(slide, "Preprocessing pipeline", { x: 100, y: 286, w: 360, h: 34, size: 24, color: colors.text, bold: true });
  addBulletList(slide, [
    "Drop customerID identifier",
    "Convert TotalCharges to numeric",
    "Encode Churn as 1/0",
    "Impute missing values",
    "Scale numeric features",
    "One-hot encode categoricals",
  ], { x: 104, y: 338, w: 394, h: 150, size: 18 });
  addShape(slide, { x: 640, y: 260, w: 494, h: 258, fill: colors.panel, line: colors.line });
  addText(slide, "Models compared", { x: 668, y: 286, w: 360, h: 34, size: 24, color: colors.text, bold: true });
  addMetricCard(slide, "LR", "Logistic Regression", { x: 670, y: 350, w: 130, h: 104, accent: colors.blue });
  addMetricCard(slide, "RF", "Random Forest", { x: 822, y: 350, w: 130, h: 104, accent: colors.teal });
  addMetricCard(slide, "GB", "Gradient Boosting", { x: 974, y: 350, w: 130, h: 104, accent: colors.yellow });
  setNotes(slide, "Preprocessing is implemented as a Scikit-learn pipeline. This is important because the same transformations are used during training and serving. I compared three baseline models before tuning.");
  script.push("## Slide 4 - Data Preparation and Models", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 5);
  addTitle(slide, "MLflow records every experiment", "Experiment tracking evidence");
  addText(slide, "Baseline, tuning, randomized search, and monitoring runs are visible in one experiment.", { x: 66, y: 218, w: 790, h: 42, size: 22, color: colors.muted });
  addScreenshotFrame(slide, "MLflow experiment run list: parameters, metrics, artifacts, and run types.", { x: 72, y: 286, w: 1038, h: 334 });
  await addImage(slide, imagePath("01_experiments_runs.png"), { x: 88, y: 302, w: 1006, h: 276, fit: "contain", alt: "MLflow experiment runs" });
  setNotes(slide, "This screenshot shows the MLflow experiment. Each run logs parameters, metrics, artifacts, and model outputs, making the comparison auditable and reproducible.");
  script.push("## Slide 5 - Experiment Tracking", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 6);
  addTitle(slide, "RandomizedSearchCV strengthens tuning", "Hyperparameter tuning results");
  addMetricCard(slide, "20", "sampled candidates", { x: 72, y: 254, w: 210, accent: colors.teal });
  addMetricCard(slide, "3-fold", "cross validation", { x: 310, y: 254, w: 210, accent: colors.blue });
  addMetricCard(slide, "0.8476", "best CV ROC-AUC", { x: 548, y: 254, w: 210, accent: colors.yellow });
  addMetricCard(slide, "0.8411", "best tuned test ROC-AUC", { x: 786, y: 254, w: 260, accent: colors.teal });
  addScreenshotFrame(slide, "Best randomized-search model run with metrics and artifacts.", { x: 72, y: 420, w: 1038, h: 198 });
  await addImage(slide, imagePath("02_best_run_metrics.png"), { x: 88, y: 436, w: 1006, h: 140, fit: "contain", alt: "Best tuning run metrics" });
  setNotes(slide, "For tuning, I used RandomizedSearchCV instead of a manual grid. It samples a broader parameter space while keeping runtime manageable. The best cross-validation ROC-AUC was about 0.8476.");
  script.push("## Slide 6 - Hyperparameter Tuning", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 7);
  addTitle(slide, "Production model is versioned", "Model registry");
  addText(slide, "Registered model: ChurnPredictionModel", { x: 66, y: 220, w: 720, h: 34, size: 24, color: colors.text, bold: true });
  addBulletList(slide, [
    "Version 1 registered in MLflow Model Registry.",
    "Production and staging aliases are assigned.",
    "Registry creates a clear handoff from experimentation to deployment.",
  ], { x: 72, y: 275, w: 480, h: 160, size: 20 });
  addScreenshotFrame(slide, "MLflow Model Registry page with production/staging aliases.", { x: 604, y: 150, w: 534, h: 446 });
  await addImage(slide, imagePath("05_production_model.png"), { x: 620, y: 166, w: 502, h: 388, fit: "contain", alt: "MLflow production model" });
  setNotes(slide, "The best model is registered as ChurnPredictionModel. The registry demonstrates model versioning and lifecycle management, including staging and production aliases.");
  script.push("## Slide 7 - Model Registry", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 8);
  addTitle(slide, "Serving and monitoring complete the loop", "Deployment and monitoring");
  addShape(slide, { x: 72, y: 250, w: 500, h: 132, fill: colors.panel, line: colors.line });
  addText(slide, "Serving command", { x: 98, y: 274, w: 220, h: 26, size: 21, color: colors.text, bold: true });
  addText(slide, "mlflow models serve -m \"models:/ChurnPredictionModel/Production\" -p 5001 --no-conda", { x: 98, y: 318, w: 430, h: 48, size: 15, color: colors.yellow, face: fonts.mono });
  addShape(slide, { x: 72, y: 410, w: 500, h: 92, fill: "#07101C", line: colors.line });
  addText(slide, "Prediction response", { x: 98, y: 430, w: 220, h: 24, size: 18, color: colors.text, bold: true });
  addText(slide, '{"predictions": [1]}', { x: 98, y: 462, w: 420, h: 26, size: 17, color: colors.teal, face: fonts.mono });
  addText(slide, "Sample customer predicted as churn risk.", { x: 98, y: 492, w: 420, h: 24, size: 14, color: colors.muted });
  addScreenshotFrame(slide, "Monitoring run: four simulated production-like batches logged to MLflow.", { x: 624, y: 246, w: 520, h: 286 });
  await addImage(slide, imagePath("03_monitoring_run.png"), { x: 640, y: 262, w: 488, h: 228, fit: "contain", alt: "Monitoring MLflow run" });
  setNotes(slide, "The production model is served locally with MLflow Models. The serve_test script sends a sample customer and receives a prediction. Monitoring then evaluates four simulated batches and logs metrics like F1-score and ROC-AUC.");
  script.push("## Slide 8 - Deployment and Monitoring", slide.speakerNotes.text, "");

  slide = newSlide(presentation, 9);
  addTitle(slide, "MLflow turns a model into a managed lifecycle", "Conclusion and demo checklist");
  addBulletList(slide, [
    "Experiment tracking makes model comparison auditable.",
    "RandomizedSearchCV makes tuning more defensible.",
    "Model Registry provides a production handoff.",
    "Serving demonstrates real-time inference.",
    "Monitoring shows how performance can be tracked over time.",
  ], { x: 76, y: 250, w: 620, h: 210, size: 21 });
  addShape(slide, { x: 780, y: 166, w: 350, h: 358, fill: colors.panel, line: colors.line });
  addText(slide, "Live demo order", { x: 810, y: 198, w: 250, h: 30, size: 24, color: colors.text, bold: true });
  addBulletList(slide, [
    "Open MLflow experiment",
    "Show best run metrics",
    "Open registry production model",
    "Show prediction response",
    "Show monitoring metrics",
  ], { x: 812, y: 252, w: 260, h: 176, size: 18 });
  addText(slide, "Repository:\ngithub.com/Ayhan-Gurbangeldiyev/AIN-3009-MLflow-Customer-Churn", { x: 812, y: 464, w: 270, h: 42, size: 13, color: colors.teal, face: fonts.mono });
  setNotes(slide, "To conclude, the project satisfies the main lifecycle requirements: tracking, training, tuning, registry, deployment, and monitoring. In the demo, I would show the MLflow experiment, best run, registry model, prediction response, and monitoring metrics.");
  script.push("## Slide 9 - Conclusion and Demo Checklist", slide.speakerNotes.text, "");

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  for (let i = 0; i < presentation.slides.count; i += 1) {
    const preview = await presentation.export({ slide: presentation.slides.getItem(i), format: "png", scale: 1 });
    const buffer = Buffer.from(await preview.arrayBuffer());
    await fs.writeFile(path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), buffer);
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  await fs.writeFile(SCRIPT_OUT, `${script.join("\n")}\n`, "utf8");
  console.log(OUT);
  console.log(SCRIPT_OUT);
}

build().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
