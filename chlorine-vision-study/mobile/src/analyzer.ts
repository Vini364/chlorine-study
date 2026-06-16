const jpeg = require('jpeg-js');
const mobileModel = require('./mobile_model.json');

type AnalyzerResultPayload = {
  predicted_concentration: number;
  unit?: string;
  model_name?: string;
  timestamp?: string;
};

type ExportedTree = {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  value: number[];
};

type ForestMobileModel = {
  model_name: string;
  model_type?: 'forest';
  unit: string;
  feature_columns: string[];
  trees: ExportedTree[];
};

type LinearMobileModel = {
  model_name: string;
  model_type: 'linear';
  unit: string;
  feature_columns: string[];
  coefficients: number[];
  intercept: number;
  scaler: {
    mean: number[];
    scale: number[];
  };
};

type MobileModel = ForestMobileModel | LinearMobileModel;

type AnalyzeInput = {
  base64: string;
};

function decodeBase64ToBytes(base64: string) {
  const normalized = base64.includes(',') ? base64.split(',').pop() || '' : base64;
  const decode =
    typeof globalThis.atob === 'function'
      ? globalThis.atob.bind(globalThis)
      : null;

  if (!decode) {
    throw new Error('Base64 decoding is not available in this build.');
  }

  const binary = decode(normalized);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function rgbToHsvOpenCv(red: number, green: number, blue: number) {
  const r = red / 255;
  const g = green / 255;
  const b = blue / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  let hueDegrees = 0;
  if (delta !== 0) {
    if (max === r) {
      hueDegrees = ((g - b) / delta) % 6;
    } else if (max === g) {
      hueDegrees = (b - r) / delta + 2;
    } else {
      hueDegrees = (r - g) / delta + 4;
    }
    hueDegrees *= 60;
    if (hueDegrees < 0) {
      hueDegrees += 360;
    }
  }

  const saturation = max === 0 ? 0 : (delta / max) * 255;
  const value = max * 255;

  return {
    h: hueDegrees / 2,
    s: saturation,
    v: value,
  };
}

function rgbToLabOpenCv(red: number, green: number, blue: number) {
  const toLinear = (channel: number) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : Math.pow((normalized + 0.055) / 1.055, 2.4);
  };

  const r = toLinear(red);
  const g = toLinear(green);
  const b = toLinear(blue);

  const x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
  const y = (r * 0.2126729 + g * 0.7151522 + b * 0.072175) / 1.0;
  const z = (r * 0.0193339 + g * 0.119192 + b * 0.9503041) / 1.08883;

  const transform = (value: number) =>
    value > 0.008856 ? Math.cbrt(value) : 7.787 * value + 16 / 116;

  const fx = transform(x);
  const fy = transform(y);
  const fz = transform(z);

  const l = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const labB = 200 * (fy - fz);

  return {
    l: (l * 255) / 100,
    a: a + 128,
    b: labB + 128,
  };
}

function computeFeatureMap(base64: string) {
  const jpegBytes = decodeBase64ToBytes(base64);
  const decoded = jpeg.decode(jpegBytes, { useTArray: true });
  const { data } = decoded;
  const pixelCount = decoded.width * decoded.height;

  if (!pixelCount) {
    throw new Error('Could not decode the cropped image for on-device analysis.');
  }

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sumR2 = 0;
  let sumG2 = 0;
  let sumB2 = 0;

  let sumH = 0;
  let sumS = 0;
  let sumV = 0;
  let sumH2 = 0;
  let sumS2 = 0;
  let sumV2 = 0;

  let sumL = 0;
  let sumA = 0;
  let sumLabB = 0;
  let sumL2 = 0;
  let sumA2 = 0;
  let sumLabB2 = 0;

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];

    sumR += red;
    sumG += green;
    sumB += blue;
    sumR2 += red * red;
    sumG2 += green * green;
    sumB2 += blue * blue;

    const hsv = rgbToHsvOpenCv(red, green, blue);
    sumH += hsv.h;
    sumS += hsv.s;
    sumV += hsv.v;
    sumH2 += hsv.h * hsv.h;
    sumS2 += hsv.s * hsv.s;
    sumV2 += hsv.v * hsv.v;

    const lab = rgbToLabOpenCv(red, green, blue);
    sumL += lab.l;
    sumA += lab.a;
    sumLabB += lab.b;
    sumL2 += lab.l * lab.l;
    sumA2 += lab.a * lab.a;
    sumLabB2 += lab.b * lab.b;
  }

  const meanR = sumR / pixelCount;
  const meanG = sumG / pixelCount;
  const meanB = sumB / pixelCount;
  const meanH = sumH / pixelCount;
  const meanS = sumS / pixelCount;
  const meanV = sumV / pixelCount;
  const meanL = sumL / pixelCount;
  const meanA = sumA / pixelCount;
  const meanBLab = sumLabB / pixelCount;

  const std = (sumSquares: number, mean: number) =>
    Math.sqrt(Math.max(sumSquares / pixelCount - mean * mean, 0));

  const total = meanR + meanG + meanB;
  const grayEstimate = total / 3;

  return {
    mean_R: meanR,
    mean_G: meanG,
    mean_B: meanB,
    std_R: std(sumR2, meanR),
    std_G: std(sumG2, meanG),
    std_B: std(sumB2, meanB),
    norm_r: total > 0 ? meanR / total : 0,
    norm_g: total > 0 ? meanG / total : 0,
    norm_b: total > 0 ? meanB / total : 0,
    ratio_R_G: meanR / (meanG + 1e-6),
    ratio_R_B: meanR / (meanB + 1e-6),
    ratio_G_B: meanG / (meanB + 1e-6),
    diff_R_G: meanR - meanG,
    diff_R_B: meanR - meanB,
    diff_G_B: meanG - meanB,
    mean_H: meanH,
    mean_S: meanS,
    mean_V: meanV,
    std_H: std(sumH2, meanH),
    std_S: std(sumS2, meanS),
    std_V: std(sumV2, meanV),
    mean_L: meanL,
    mean_A: meanA,
    mean_B_lab: meanBLab,
    std_L: std(sumL2, meanL),
    std_A: std(sumA2, meanA),
    std_B_lab: std(sumLabB2, meanBLab),
    brightness: total / 3,
    color_saturation: Math.sqrt(
      Math.pow(meanR - grayEstimate, 2) +
        Math.pow(meanG - grayEstimate, 2) +
        Math.pow(meanB - grayEstimate, 2)
    ),
    dominant_channel: meanR >= meanG && meanR >= meanB ? 0 : meanG >= meanB ? 1 : 2,
  };
}

function predictTree(tree: ExportedTree, featureVector: number[]) {
  let nodeIndex = 0;

  while (tree.children_left[nodeIndex] !== -1 && tree.children_right[nodeIndex] !== -1) {
    const featureIndex = tree.feature[nodeIndex];
    const threshold = tree.threshold[nodeIndex];
    nodeIndex =
      featureVector[featureIndex] <= threshold
        ? tree.children_left[nodeIndex]
        : tree.children_right[nodeIndex];
  }

  return tree.value[nodeIndex];
}

function predictWithForest(model: MobileModel, features: Record<string, number>) {
  const featureVector = model.feature_columns.map((name) => features[name] ?? 0);
  const forestModel = model as ForestMobileModel;
  const sum = forestModel.trees.reduce(
    (accumulator, tree) => accumulator + predictTree(tree, featureVector),
    0
  );
  return sum / forestModel.trees.length;
}

function predictWithLinearModel(model: LinearMobileModel, features: Record<string, number>) {
  return model.feature_columns.reduce((prediction, featureName, index) => {
    const rawValue = features[featureName] ?? 0;
    const mean = model.scaler.mean[index] ?? 0;
    const scale = model.scaler.scale[index] ?? 1;
    const standardizedValue = (rawValue - mean) / (scale || 1);
    return prediction + standardizedValue * (model.coefficients[index] ?? 0);
  }, model.intercept);
}

function predictConcentration(model: MobileModel, features: Record<string, number>) {
  if (model.model_type === 'linear') {
    return predictWithLinearModel(model, features);
  }

  return predictWithForest(model, features);
}

export async function checkAnalyzerReady() {
  return;
}

export async function analyzeSample(input: AnalyzeInput): Promise<AnalyzerResultPayload> {
  if (!input.base64) {
    throw new Error('Missing image data for on-device analysis.');
  }

  const features = computeFeatureMap(input.base64);
  const model = mobileModel as MobileModel;
  const predictedConcentration = Math.max(0, predictConcentration(model, features));

  return {
    predicted_concentration: predictedConcentration,
    model_name: model.model_name,
    timestamp: new Date().toISOString(),
    unit: model.unit || 'mg/L',
  };
}

export function getAnalyzerUnavailableMessage() {
  return 'On-device analysis is unavailable in this build.';
}
