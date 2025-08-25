import { env } from 'onnxruntime-web';
import type { Tensor } from 'onnxruntime-web';
import { SAMEncoder } from './encoder';
import { SAMDecoder } from './decoder';
import type { DecodeOptions } from './types';

class SAM {
  public encoder: SAMEncoder;
  public decoder: SAMDecoder;

  public constructor() {
    const encoderConfig = {
      targetSize: 1024,
      modelPath: '/models/sam2_image_encoder_1024.onnx',
    };
    const decoderConfig = {
      targetSize: 1024,
      modelPath: '/models/sam2_mask_decoder_1024.onnx',
    };

    this.encoder = new SAMEncoder(encoderConfig);
    this.decoder = new SAMDecoder(decoderConfig);
  }

  async init() {
    /** 初始化 onnxruntime-web */
    env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    env.wasm.proxy = true;

    /** 初始化 encoder 和 decoder */
    await this.encoder.init();
    await this.decoder.init();
  }

  async prepare(image: HTMLImageElement) {
    return this.encoder.prepare(image);
  }

  async encode(imageTensor: Tensor) {
    return this.encoder.encode(imageTensor);
  }

  async decode(options: DecodeOptions) {
    return this.decoder.decode(options);
  }
}

export { SAM };
