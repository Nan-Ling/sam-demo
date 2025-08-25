import { InferenceSession, Tensor } from 'onnxruntime-web';
import type { DecoderConfig, DecodeOptions, DecodeResult, DecodeMask, DecodeLowResMask } from './types';

const DEFAULT_COLOR = [0, 114, 189, 255];

class SAMDecoder {
  private modelPath: string;
  private session: InferenceSession | null = null;

  public constructor(config: DecoderConfig) {
    this.modelPath = config.modelPath;
  }

  public async init() {
    this.session = await InferenceSession.create(this.modelPath);
  }

  private processIouPredictions(iouPredictionsTensor: Tensor) {
    const [, count] = iouPredictionsTensor.dims;
    const iouPredictionsData = iouPredictionsTensor.data as Float32Array;
    const iouPredictions: number[] = [];
    for (let i = 0; i < count; i++) {
      const percent = Number(iouPredictionsData[i] * 100).toFixed(2);
      iouPredictions.push(Number(percent));
    }

    return iouPredictions;
  }

  private processMasks(masksTensor: Tensor, iouPredictions: number[]) {
    const [, count, height, width] = masksTensor.dims;
    const masksData = masksTensor.data as Uint8Array;
    const size = height * width;
    const [r, g, b, a] = DEFAULT_COLOR;

    const masks: ImageData[] = [];
    for (let i = 0; i < count; i++) {
      const mask = masksData.slice(i * size, (i + 1) * size);
      const imageData = new ImageData(width, height);
      for (let j = 0; j < size; j++) {
        if (mask[j] > 0.0) {
          imageData.data[j * 4 + 0] = r;
          imageData.data[j * 4 + 1] = g;
          imageData.data[j * 4 + 2] = b;
          imageData.data[j * 4 + 3] = a;
        }
      }

      masks.push(imageData);
    }

    const maskResults: DecodeMask[] = [];
    for (let i = 0; i < count; i++) {
      const mask: DecodeMask = {
        imageData: masks[i],
        iouPrediction: iouPredictions[i],
      };
      maskResults.push(mask);
    }

    return maskResults;
  }

  private processLowResMasks(lowResMasksTensor: Tensor, iouPredictions: number[]) {
    const [, count, height, width] = lowResMasksTensor.dims;
    const lowResMasksData = lowResMasksTensor.data as Float32Array;
    const size = height * width;
    const [r, g, b, a] = DEFAULT_COLOR;

    const lowResMasks: DecodeLowResMask[] = [];
    for (let i = 0; i < count; i++) {
      const imageData = new ImageData(width, height);
      const data = new Float32Array(lowResMasksData.slice(i * size, (i + 1) * size));
      for (let j = 0; j < size; j++) {
        if (data[j] > 0.0) {
          imageData.data[j * 4 + 0] = r;
          imageData.data[j * 4 + 1] = g;
          imageData.data[j * 4 + 2] = b;
          imageData.data[j * 4 + 3] = a;
        }
      }

      const lowResMaskResult: DecodeLowResMask = {
        imageData,
        data,
        iouPrediction: iouPredictions[i],
      };
      lowResMasks.push(lowResMaskResult);
    }

    return lowResMasks;
  }

  public async decode(options: DecodeOptions) {
    const session = this.session;
    if (!session) {
      throw new Error('Session not initialized');
    }

    const numPoints = options.pointCoords.length;
    const pointCoords = new Float32Array(options.pointCoords.flatMap(point => [point.x, point.y]));
    const pointLabels = new BigInt64Array(options.pointLabels.map(point => BigInt(point.label)));
    const boxes = new Float32Array([
      options.boxes.topLeft.x,
      options.boxes.topLeft.y,
      options.boxes.bottomRight.x,
      options.boxes.bottomRight.y,
    ]);
    const maskInput = new Float32Array(options.maskInput);

    /** DecodeModelFeeds */
    const decoderFeeds = {
      image_embed: new Tensor('float32', new Float32Array(options.imageEmbed), [1, 256, 64, 64]),
      high_res_feat1: new Tensor('float32', new Float32Array(options.highResFeat1), [1, 32, 256, 256]),
      high_res_feat2: new Tensor('float32', new Float32Array(options.highResFeat2), [1, 64, 128, 128]),
      point_coords: new Tensor('float32', pointCoords, [1, numPoints, 2]),
      point_labels: new Tensor('int64', pointLabels, [1, numPoints]),
      boxes: new Tensor('float32', boxes, [1, 4]),
      mask_input: new Tensor('float32', maskInput, [1, 1, 256, 256]),
    };

    /** DecodeModelResult */
    const result = await session.run(decoderFeeds);

    console.log(result);

    const iouPredictions = this.processIouPredictions(result.iou_predictions);
    const masks = this.processMasks(result.masks, iouPredictions);
    const lowResMasks = this.processLowResMasks(result.low_res_masks, iouPredictions);

    const exportedResult: DecodeResult = {
      masks,
      iouPredictions,
      lowResMasks,
    };

    return exportedResult;
  }
}

export { SAMDecoder };
