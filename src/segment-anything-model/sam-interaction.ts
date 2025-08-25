import { SAM } from './sam';
import { resizeImageDataToTargetSizeImage, imageDataToCanvas } from './utils';
import type { EncodeResult, Point, LabelPoint, DecodeOptions, Box, DecodeResult, ResizeImageResult } from './types';

class SAMInteraction {
  private sam: SAM;
  public image: HTMLImageElement;
  private encodeCache: EncodeResult | null;
  public resizeImageResult: ResizeImageResult | null;

  private pointCoords: Point[];
  private pointLabels: LabelPoint[];
  private boxes: Box;
  private maskInput: Float32Array;

  private previousDecodeResult: DecodeResult | null;

  public constructor(sam: SAM, image: HTMLImageElement) {
    this.sam = sam;
    this.image = image;

    this.encodeCache = null;
    this.resizeImageResult = null;

    this.pointCoords = [];
    this.pointLabels = [];
    this.boxes = {
      topLeft: { x: 0, y: 0 },
      bottomRight: { x: 0, y: 0 },
    };
    this.maskInput = new Float32Array(256 * 256).fill(0);

    this.previousDecodeResult = null;
  }

  public setEncodeCache(encodeCache: EncodeResult, resizeImageResult: ResizeImageResult) {
    this.encodeCache = encodeCache;
    this.resizeImageResult = resizeImageResult;
  }

  public setPoints(points: LabelPoint[]) {
    this.pointLabels = points;
    this.pointCoords = points.map(point => ({ x: point.x, y: point.y }));
  }

  public addPoint(point: LabelPoint) {
    this.pointLabels.push(point);
    this.pointCoords.push({ x: point.x, y: point.y });
  }

  public setBox(box: Box) {
    this.boxes = box;
  }

  public setMaskInput(maskInput: Float32Array) {
    this.maskInput = maskInput;
  }

  public mapOriginalToCanvasCoords(point: Point) {
    if (!this.resizeImageResult) {
      throw new Error('resizeImageResult 不存在，请先调用 prepare 方法');
    }

    const { mapOriginalToCanvasCoords } = this.resizeImageResult;

    return mapOriginalToCanvasCoords(point.x, point.y);
  }

  public async prepare() {
    if (this.encodeCache) {
      console.warn('encodeCache 已存在，跳过编码');
      return this.encodeCache;
    }

    const { imageTensor, ...resizeImageResult } = await this.sam.prepare(this.image);
    this.resizeImageResult = resizeImageResult;
    this.encodeCache = await this.sam.encode(imageTensor);

    return this.encodeCache;
  }

  private createDecodeOptions(pointCoords: Point[], pointLabels: LabelPoint[], boxes: Box, maskInput: Float32Array) {
    if (!this.encodeCache) {
      throw new Error('encodeCache 不存在，请先调用 prepare 方法');
    }

    const options: DecodeOptions = {
      imageEmbed: this.encodeCache.imageEmbed,
      highResFeat1: this.encodeCache.highResFeat1,
      highResFeat2: this.encodeCache.highResFeat2,
      pointCoords,
      pointLabels,
      boxes,
      maskInput,
    };

    return options;
  }

  public showMasks(result: DecodeResult) {
    if (!this.resizeImageResult) {
      throw new Error('resizeImageResult 不存在，请先调用 prepare 方法');
    }

    for (const mask of result.masks) {
      const canvas = resizeImageDataToTargetSizeImage({
        imageData: mask.imageData,
        targetWidth: this.resizeImageResult.originalWidth,
        targetHeight: this.resizeImageResult.originalHeight,
        offsetX: this.resizeImageResult.offsetX,
        offsetY: this.resizeImageResult.offsetY,
        viewWidth: this.resizeImageResult.scaledWidth,
        viewHeight: this.resizeImageResult.scaledHeight,
      });
    }

    for (const mask of result.lowResMasks) {
      const canvas = imageDataToCanvas(mask.imageData);
    }
  }

  public getDecodeOptions() {
    return this.createDecodeOptions(this.pointCoords, this.pointLabels, this.boxes, this.maskInput);
  }

  public async decode(options: DecodeOptions) {
    if (!this.resizeImageResult) {
      throw new Error('resizeImageResult 不存在，请先调用 prepare 方法');
    }

    const result = await this.sam.decode(options);

    this.previousDecodeResult = result;

    const masks: HTMLCanvasElement[] = [];

    for (const mask of result.masks) {
      const canvas = resizeImageDataToTargetSizeImage({
        imageData: mask.imageData,
        targetWidth: this.resizeImageResult.originalWidth,
        targetHeight: this.resizeImageResult.originalHeight,
        offsetX: this.resizeImageResult.offsetX,
        offsetY: this.resizeImageResult.offsetY,
        viewWidth: this.resizeImageResult.scaledWidth,
        viewHeight: this.resizeImageResult.scaledHeight,
      });
      masks.push(canvas);
    }

    return {
      masks,
      result,
    };
  }

  public async refine() {
    if (!this.previousDecodeResult) {
      throw new Error('previousDecodeResult 不存在，请先调用 decode 方法');
    }

    const maskInput = new Float32Array(this.previousDecodeResult.lowResMasks[0].data);
    const options = this.createDecodeOptions(this.pointCoords, this.pointLabels, this.boxes, maskInput);
    const { result, masks } = await this.decode(options);

    this.previousDecodeResult = result;

    return {
      masks,
    };
  }
}

export { SAMInteraction };
