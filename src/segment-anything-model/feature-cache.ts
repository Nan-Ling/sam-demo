interface CachedTensor {
  buffer: ArrayBuffer;
}

class FeatureCache {
  private readonly databaseName: string;
  private readonly objectStoreName: string;

  constructor(databaseName = 'sam2-cache', objectStoreName = 'features') {
    this.databaseName = databaseName;
    this.objectStoreName = objectStoreName;
  }

  private openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.databaseName, 1);
      request.onupgradeneeded = event => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.objectStoreName)) {
          db.createObjectStore(this.objectStoreName);
        }
      };
      request.onsuccess = event => {
        resolve((event.target as IDBOpenDBRequest).result);
      };
      request.onerror = event => {
        reject((event.target as IDBOpenDBRequest).error);
      };
    });
  }

  public async save(key: string, data: Float32Array): Promise<void> {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.objectStoreName, 'readwrite');
      const store = tx.objectStore(this.objectStoreName);
      const cached: CachedTensor = {
        buffer: data,
      };
      store.put(cached, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  public async load(key: string): Promise<Float32Array | null> {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.objectStoreName, 'readonly');
      const store = tx.objectStore(this.objectStoreName);
      const req = store.get(key);
      req.onsuccess = () => {
        const cached = req.result as CachedTensor | undefined;
        if (cached) {
          const array = new Float32Array(cached.buffer);
          resolve(array);
        } else {
          resolve(null);
        }
      };
      req.onerror = () => reject(req.error);
    });
  }
}

export { FeatureCache };
