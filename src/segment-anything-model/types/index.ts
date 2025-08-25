import type { Tensor } from 'onnxruntime-web';

interface EncoderConfig {
  targetSize: number;
  modelPath: string;
}

interface DecoderConfig {
  targetSize: number;
  modelPath: string;
}

/** 编码模型输入 */
interface EncodeModelFeeds {
  /**
   * 将 ImageData 转为 unit8 的 tensor，用于编码模型输入
   * tensor 的 shape 为 [1, 3, 1024, 1024]
   * ImageData 的数据格式是 [r, g, b, a]
   * tensor 的数据格式为 [r,r,r,r,g,g,g,g,b,b,b,b,a,a,a,a]
   * tensor 存储数据是按照通道存储的，所以需要将 ImageData 的数据格式转换为 tensor 的数据格式
   */
  image: Tensor;
}

/** 编码模型输出 */
interface EncodeModelResult {
  /** 图片特征 */
  image_embed: Tensor;
  /** 高分辨率特征1 */
  high_res_feat1: Tensor;
  /** 高分辨率特征2 */
  high_res_feat2: Tensor;
}

/** 编码器对外输出，只是将名称改成驼峰命名 */
interface EncodeResult {
  imageEmbed: Float32Array;
  highResFeat1: Float32Array;
  highResFeat2: Float32Array;
}

/** 解码模型输入 */
interface DecodeModelFeeds {
  /** 图片特征 */
  image_embed: Tensor;
  /** 高分辨率特征1 */
  high_res_feat1: Tensor;
  /** 高分辨率特征2 */
  high_res_feat2: Tensor;
  /**
   * 点坐标
   * tensor 的 shape 为 [1, numPoints, 2]
   * 数据格式为 [x1, y1, x2, y2, ...]
   * 其中 x1, y1 是第一个点的坐标，x2, y2 是第二个点的坐标，以此类推
   * 数据类型为 float32
   */
  point_coords: Tensor;
  /**
   * 点标签
   * tensor 的 shape 为 [1, numPoints]
   * 数据格式为 [0, 1, 0, 1, ...]
   * 其中 0 表示背景，1 表示前景
   * 数据类型为 int64
   * 这里的一个标签对应一个点坐标，即 x1, y1 对应一个标签，x2, y2 对应一个标签，以此类推
   */
  point_labels: Tensor;
  /**
   * 框坐标
   * tensor 的 shape 为 [1, 4]
   * 数据格式为 [x1, y1, x2, y2]
   * 其中 x1, y1 是框的左上角坐标，x2, y2 是框的右下角坐标
   * 数据类型为 float32
   */
  boxes: Tensor;
  /** 上一次推理返回的 low_res_masks，用于后续的优化 */
  mask_input: Tensor;
}

/** 解码模型输出 */
interface DecodeModelResult {
  /** 掩码 */
  masks: Tensor;
  /** 掩码分数 */
  iou_predictions: Tensor;
  /** 低分辨率掩码，用于后续的优化 */
  low_res_masks: Tensor;
}

interface Box {
  topLeft: Point;
  bottomRight: Point;
}

interface Point {
  x: number;
  y: number;
}

interface LabelPoint extends Point {
  label: number;
}

interface DecodeOptions extends EncodeResult {
  pointCoords: Point[];
  pointLabels: LabelPoint[];
  boxes: Box;
  maskInput: Float32Array;
}

interface DecodeLowResMask {
  imageData: ImageData;
  data: Float32Array;
  iouPrediction: number;
}

interface DecodeMask {
  imageData: ImageData;
  iouPrediction: number;
}

/** 解码器对外输出，只是将名称改成驼峰命名 */
interface DecodeResult {
  masks: DecodeMask[];
  iouPredictions: number[];
  lowResMasks: DecodeLowResMask[];
}

type MapOriginalToCanvasCoords = (x: number, y: number) => Point;

interface ResizeImageResult {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  scale: number;
  originalWidth: number;
  originalHeight: number;
  scaledWidth: number;
  scaledHeight: number;
  offsetX: number;
  offsetY: number;
  mapOriginalToCanvasCoords: MapOriginalToCanvasCoords;
}

export type {
  EncoderConfig,
  DecoderConfig,
  EncodeResult,
  EncodeModelFeeds,
  EncodeModelResult,
  DecodeModelFeeds,
  DecodeModelResult,
  DecodeOptions,
  Point,
  LabelPoint,
  Box,
  DecodeResult,
  DecodeMask,
  DecodeLowResMask,
  MapOriginalToCanvasCoords,
  ResizeImageResult,
};
