import { Tensor } from 'onnxruntime-web';
import type { MapOriginalToCanvasCoords, ResizeImageResult } from '../types';

/**
 * 创建canvas
 * @param width 宽度
 * @param height 高度
 */
function createCanvas(width: number, height: number) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Failed to get canvas context');
  }

  return { canvas, context };
}

/**
 * 将 canvas 转换为 image
 */
function canvasToImage(canvas: HTMLCanvasElement) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    canvas.toBlob(blob => {
      if (!blob) {
        reject(new Error('Failed to convert canvas to blob'));
        return;
      }

      const image = new Image();
      image.src = URL.createObjectURL(blob);

      image.addEventListener('load', () => {
        URL.revokeObjectURL(image.src);
        resolve(image);
      });

      image.addEventListener('error', () => {
        reject(new Error('Failed to convert canvas to image'));
      });
    });
  });
}

/**
 * 将图片缩放+padding到目标尺寸
 * @param image 原图
 * @param targetSize 目标尺寸
 */
function resizeImageToTargetSize(image: HTMLImageElement, targetSize: number) {
  const originalWidth = image.naturalWidth;
  const originalHeight = image.naturalHeight;
  /** 长边缩放 */
  const scale = targetSize / Math.max(originalWidth, originalHeight);
  const scaledWidth = Math.round(originalWidth * scale);
  const scaledHeight = Math.round(originalHeight * scale);

  const { canvas, context } = createCanvas(targetSize, targetSize);
  context.fillStyle = 'black';
  context.fillRect(0, 0, targetSize, targetSize);

  /** 居中绘制缩放后的图片 */
  const offsetX = Math.floor((targetSize - scaledWidth) / 2);
  const offsetY = Math.floor((targetSize - scaledHeight) / 2);
  context.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);

  const mapPoint: MapOriginalToCanvasCoords = (x, y) => {
    const mappedX = x * scale + offsetX;
    const mappedY = y * scale + offsetY;

    return { x: mappedX, y: mappedY };
  };

  const result: ResizeImageResult = {
    canvas,
    context,
    scale,
    originalWidth,
    originalHeight,
    scaledWidth,
    scaledHeight,
    offsetX,
    offsetY,
    mapOriginalToCanvasCoords: mapPoint,
  };

  return result;
}

/**
 * 将 canvas 图像数据转为 Uint8Tensor
 * @param context 画布上下文
 * @param targetSize 目标尺寸
 */
function imageDataToUint8Tensor(context: CanvasRenderingContext2D, targetSize: number): Tensor {
  const imageData = context.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData;

  const size = targetSize * targetSize;
  const uint8Data = new Uint8Array(size * 3);

  for (let i = 0; i < size; i++) {
    uint8Data[i] = data[i * 4];
    uint8Data[i + size] = data[i * 4 + 1];
    uint8Data[i + size * 2] = data[i * 4 + 2];
  }

  return new Tensor('uint8', uint8Data, [1, 3, targetSize, targetSize]);
}

/**
 * 将 imageData 转换为 canvas
 * @param imageData 图像数据
 * @returns canvas
 */
function imageDataToCanvas(imageData: ImageData) {
  const { canvas, context } = createCanvas(imageData.width, imageData.height);
  context.putImageData(imageData, 0, 0);

  return canvas;
}

interface ResizeImageDataOptions {
  imageData: ImageData;
  targetWidth: number;
  targetHeight: number;
  offsetX: number;
  offsetY: number;
  viewWidth: number;
  viewHeight: number;
}
/**
 * 将 imageData 转换为指定尺寸的 canvas
 * @param options 选项
 * @returns canvas
 */
function resizeImageDataToTargetSizeImage(options: ResizeImageDataOptions) {
  const { imageData, targetWidth, targetHeight, offsetX, offsetY, viewWidth, viewHeight } = options;
  const targetCanvas = createCanvas(targetWidth, targetHeight);
  const cropCanvas = imageDataToCanvas(imageData);

  const targetContext = targetCanvas.context;
  targetContext.drawImage(cropCanvas, offsetX, offsetY, viewWidth, viewHeight, 0, 0, targetWidth, targetHeight);

  return targetCanvas.canvas;
}

const debugCanvasElement = document.createElement('div');
document.body.appendChild(debugCanvasElement);

function debugCanvas(canvas: HTMLImageElement | HTMLCanvasElement) {
  canvas.style.width = '200px';
  canvas.style.margin = '20px';
  debugCanvasElement.insertBefore(canvas, debugCanvasElement.firstChild);
}

export {
  createCanvas,
  resizeImageToTargetSize,
  imageDataToUint8Tensor,
  canvasToImage,
  imageDataToCanvas,
  resizeImageDataToTargetSizeImage,
  debugCanvas,
};
