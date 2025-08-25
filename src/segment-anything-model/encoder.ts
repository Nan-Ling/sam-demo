import { InferenceSession } from 'onnxruntime-web';
import type { Tensor } from 'onnxruntime-web';
import { resizeImageToTargetSize, imageDataToUint8Tensor } from './utils';
import type { EncoderConfig, EncodeResult } from './types';

class SAMEncoder {
  private targetSize: number;
  private modelPath: string;
  private session: InferenceSession | null = null;

  public constructor(config: EncoderConfig) {
    this.targetSize = config.targetSize;
    this.modelPath = config.modelPath;
  }

  public async init() {
    this.session = await InferenceSession.create(this.modelPath);
  }

  public async prepare(image: HTMLImageElement) {
    const targetSize = this.targetSize;
    const resizeImageResult = resizeImageToTargetSize(image, targetSize);
    const imageTensor = imageDataToUint8Tensor(resizeImageResult.context, targetSize);

    return {
      imageTensor,
      ...resizeImageResult,
    };
  }

  public async encode(imageTensor: Tensor) {
    const session = this.session;
    if (!session) {
      throw new Error('Session not initialized');
    }

    /** EncodeModelFeeds */
    const encoderFeeds = {
      image: imageTensor,
    };

    /** EncodeModelResult */
    const result = await session.run(encoderFeeds);

    const exportedResult: EncodeResult = {
      imageEmbed: result.image_embed.data as Float32Array,
      highResFeat1: result.high_res_feat1.data as Float32Array,
      highResFeat2: result.high_res_feat2.data as Float32Array,
    };

    return exportedResult;
  }
}

export { SAMEncoder };
