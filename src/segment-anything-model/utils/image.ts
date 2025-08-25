function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.src = src;
    image.crossOrigin = 'anonymous';

    image.addEventListener('load', () => resolve(image));

    image.addEventListener('error', () => reject(new Error('Failed to load image')));
  });
}

export { loadImage };
