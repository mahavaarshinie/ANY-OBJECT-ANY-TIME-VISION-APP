import { NextRequest, NextResponse } from 'next/server';
import * as ort from 'onnxruntime-node';
import { createCanvas, loadImage } from 'canvas';
import path from 'path';

// Core Distribution Parameters
let referenceMean: number[] = [];
let referenceVariance: number[] = [];
let calibratedThreshold = 3.0;
let modelSession: ort.InferenceSession | null = null;

async function preprocessImage(fileBlob: Blob): Promise<Float32Array> {
  const buffer = Buffer.from(await fileBlob.arrayBuffer());
  const image = await loadImage(buffer);
  
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 224, 224);

  const imgData = ctx.getImageData(0, 0, 224, 224);
  const float32Data = new Float32Array(1 * 3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    const r = imgData.data[i * 4] / 255.0;
    const g = imgData.data[i * 4 + 1] / 255.0;
    const b = imgData.data[i * 4 + 2] / 255.0;

    float32Data[i] = (r - mean[0]) / std[0];
    float32Data[224 * 224 + i] = (g - mean[1]) / std[1];
    float32Data[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
  }
  return float32Data;
}

async function extractFeatures(float32Data: Float32Array): Promise<Float32Array> {
  if (!modelSession) {
    const modelPath = path.join(process.cwd(), 'public', 'models', 'wide_resnet50.onnx');
    modelSession = await ort.InferenceSession.create(modelPath);
  }
  const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  const outputs = await modelSession.run({ input: inputTensor });
  return outputs.layer2.data as Float32Array;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const mode = formData.get('mode') as string;
    const [channels, h, w] = [512, 28, 28];

    // ==========================================
    // MODE A: STATISTICAL VARIANCE DISTRIBUTION TRAINING
    // ==========================================
    if (mode === 'train') {
      const files = formData.getAll('files') as Blob[];
      const validFiles = files.filter(f => f instanceof Blob && f.size > 0);

      if (validFiles.length < 2) {
        return NextResponse.json({ error: 'Please upload AT LEAST 2 different photos of clean chips to calibrate variance calculations.' }, { status: 400 });
      }

      // Gather global pooled feature representations for all reference files
      const signatures: number[][] = [];
      for (const file of validFiles) {
        const preprocessed = await preprocessImage(file);
        const l2Data = await extractFeatures(preprocessed);
        
        const pooled = new Array(channels).fill(0);
        for (let c = 0; c < channels; c++) {
          let sum = 0;
          for (let idx = 0; idx < h * w; idx++) sum += l2Data[c * (h * w) + idx];
          pooled[c] = sum / (h * w);
        }
        signatures.push(pooled);
      }

      // Calculate mean per channel
      referenceMean = new Array(channels).fill(0);
      for (let c = 0; c < channels; c++) {
        let sum = 0;
        for (let i = 0; i < signatures.length; i++) sum += signatures[i][c];
        referenceMean[c] = sum / signatures.length;
      }

      // Calculate variance per channel (add an epsilon to avoid division by zero)
      referenceVariance = new Array(channels).fill(0);
      const epsilon = 1e-4;
      for (let c = 0; c < channels; c++) {
        let sumSquares = 0;
        for (let i = 0; i < signatures.length; i++) {
          const diff = signatures[i][c] - referenceMean[c];
          sumSquares += diff * diff;
        }
        referenceVariance[c] = (sumSquares / signatures.length) + epsilon;
      }

      // Benchmark internal training distance to calibrate decision boundary
      let maxTrainDist = 0;
      for (const sig of signatures) {
        let distance = 0;
        for (let c = 0; c < channels; c++) {
          const diff = sig[c] - referenceMean[c];
          distance += (diff * diff) / referenceVariance[c]; // Variance scaled distance
        }
        const finalDist = Math.sqrt(distance);
        if (finalDist > maxTrainDist) maxTrainDist = finalDist;
      }

      calibratedThreshold = maxTrainDist * 1.5; // 50% buffer above normal variation
      return NextResponse.json({ success: true, registered_patches: validFiles.length });
    }

    // ==========================================
    // MODE B: ROBUST VARIANCE-SCALED INFERENCE
    // ==========================================
    if (mode === 'test') {
      const file = formData.get('file') as Blob;
      if (!file || !(file instanceof Blob) || file.size === 0) {
        return NextResponse.json({ error: 'Valid testing image file target is required.' }, { status: 400 });
      }
      if (referenceMean.length === 0) {
        return NextResponse.json({ error: 'Train the model with at least 2 clean samples first.' }, { status: 400 });
      }

      const preprocessed = await preprocessImage(file);
      const l2Data = await extractFeatures(preprocessed);

      // Extract pooled test signature
      const testSig = new Array(channels).fill(0);
      for (let c = 0; c < channels; c++) {
        let sum = 0;
        for (let idx = 0; idx < h * w; idx++) sum += l2Data[c * (h * w) + idx];
        testSig[c] = sum / (h * w);
      }

      // Evaluate variance-normalized distance (Mahalanobis proxy)
      let testDistance = 0;
      for (let c = 0; c < channels; c++) {
        const diff = testSig[c] - referenceMean[c];
        testDistance += (diff * diff) / referenceVariance[c]; 
      }
      const finalAnomalyScore = Math.sqrt(testDistance);

      return NextResponse.json({
        anomaly_score: finalAnomalyScore,
        defect_detected: finalAnomalyScore > calibratedThreshold,
        threshold_limit: calibratedThreshold,
      });
    }

    return NextResponse.json({ error: 'Invalid configuration mode.' }, { status: 400 });

  } catch (error: any) {
    return NextResponse.json({ error: `Pipeline Exception: ${error.message}` }, { status: 500 });
  }
}