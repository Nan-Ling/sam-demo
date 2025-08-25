<template>
  <div class="stage">
    <canvas ref="canvasRef"></canvas>
    <p>Mouse position: x: {{ mousePosition.x.toFixed(2) }}, y: {{ mousePosition.y.toFixed(2) }}</p>
    <button ref="refineButtonRef">Refine</button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import {
  loadImage,
  SAMInteraction,
  SAM,
  resizeImageToTargetSize,
  FeatureCache,
  createCanvas,
  debugCanvas,
} from '../segment-anything-model';

const canvasRef = ref<HTMLCanvasElement | null>(null);
const mousePosition = ref({ x: 0, y: 0 });
const image = ref<HTMLImageElement | null>(null);
const refineButtonRef = ref<HTMLButtonElement | null>(null);

const MAX_CANVAS_SIZE = 800;

onMounted(async () => {
  const canvas = canvasRef.value;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  /**
   * 如果更换了图片，需要清空 IndexedDB 中的缓存，否则会使用错误的缓存
   */
  const loadedImage = await loadImage('/images/9a8abb94d1748fd06d79199c09a6b7b83277.jpeg');
  image.value = loadedImage;

  const { width, height } = loadedImage;
  const longerSide = Math.max(width, height);
  const scale = MAX_CANVAS_SIZE / longerSide;

  canvas.width = width * scale;
  canvas.height = height * scale;

  ctx.drawImage(loadedImage, 0, 0, canvas.width, canvas.height);

  const sam = new SAM();
  await sam.init();
  const samInteraction = new SAMInteraction(sam, loadedImage);

  const featureCache = new FeatureCache();
  let image_embed = await featureCache.load('image_embed');
  let high_res_feat1 = await featureCache.load('high_res_feat1');
  let high_res_feat2 = await featureCache.load('high_res_feat2');
  if (!image_embed || !high_res_feat1 || !high_res_feat2) {
    const encodeCache = await samInteraction.prepare();
    image_embed = encodeCache.imageEmbed;
    high_res_feat1 = encodeCache.highResFeat1;
    high_res_feat2 = encodeCache.highResFeat2;
    await featureCache.save('image_embed', image_embed);
    await featureCache.save('high_res_feat1', high_res_feat1);
    await featureCache.save('high_res_feat2', high_res_feat2);
  }

  console.log('解码完成');

  const resizeImageResult = resizeImageToTargetSize(loadedImage, 1024);
  samInteraction.setEncodeCache(
    {
      imageEmbed: image_embed,
      highResFeat1: high_res_feat1,
      highResFeat2: high_res_feat2,
    },
    resizeImageResult
  );

  let isDecoding = false;

  const handleMouseClick = async (event: MouseEvent) => {
    const isShiftKey = event.shiftKey;

    const rect = canvas.getBoundingClientRect();

    mousePosition.value = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    const canvasPoint = samInteraction.mapOriginalToCanvasCoords({
      x: mousePosition.value.x / scale,
      y: mousePosition.value.y / scale,
    });

    samInteraction.addPoint({ x: canvasPoint.x, y: canvasPoint.y, label: isShiftKey ? 0 : 1 });
    if (isShiftKey) {
      ctx.fillStyle = 'red';
    } else {
      ctx.fillStyle = 'blue';
    }
    ctx.fillRect(mousePosition.value.x - 5, mousePosition.value.y - 5, 10, 10);

    if (isDecoding) {
      return;
    }

    isDecoding = true;
    console.time('decode');
    const decodeResult = await samInteraction.decode(samInteraction.getDecodeOptions());
    console.timeEnd('decode');
    for (const mask of decodeResult.masks) {
      const canvas = createCanvas(samInteraction.image.naturalWidth, samInteraction.image.naturalHeight);
      canvas.context.drawImage(samInteraction.image, 0, 0);
      canvas.context.globalCompositeOperation = 'destination-in';
      canvas.context.drawImage(mask, 0, 0);

      debugCanvas(canvas.canvas);
    }

    isDecoding = false;
  };

  const handleRefine = async () => {
    const decodeResult = await samInteraction.refine();
    for (const mask of decodeResult.masks) {
      const canvas = createCanvas(samInteraction.image.naturalWidth, samInteraction.image.naturalHeight);
      canvas.context.drawImage(samInteraction.image, 0, 0);
      canvas.context.globalCompositeOperation = 'destination-in';
      canvas.context.drawImage(mask, 0, 0);

      debugCanvas(canvas.canvas);
    }
  };

  canvas.addEventListener('click', handleMouseClick);
  refineButtonRef.value?.addEventListener('click', handleRefine);
});
</script>

<style scoped>
.stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 100px auto 0;
}

canvas {
  border: 1px solid black;
}
</style>
