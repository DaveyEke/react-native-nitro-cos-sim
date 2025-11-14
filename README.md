# React Native Nitro Cosine Similarity

Fast cosine similarity for React Native, implemented in Swift using [Nitro Modules](https://github.com/mrousavy/nitro).

- **Supports**: JS arrays and `ArrayBuffer` / typed arrays  
- **Optimized for**: OpenAI‑style embedding vectors (e.g. length 1536)  
- **Bridge**: Nitro Modules (zero‑copy for `ArrayBuffer`)

| Implementation                            | Notes                                              |
| ----------------------------------------- | -------------------------------------------------- |
| Swift (ArrayBuffer, scalar, no vDSP)      | Zero‑copy, pointer‑based, Release build optimized |
| JavaScript (Vercel AI `cosineSimilarity`) | Baseline, JIT‑optimized JS                         |

This project explores different approaches to implementing cosine similarity in React Native, with a focus on performance through native code and zero‑copy memory access.

![ArrayBuffer Demo](docs/images/ArrayBufferDemo.png)  
*Native implementation with ArrayBuffer (zero‑copy) – older benchmark*

![Array Demo](docs/images/ArrayDemo.png)  
*Native implementation with regular arrays – shows marshalling overhead*

Latest scalar ArrayBuffer + Release benchmark:

![Scalar Swift with ArrayBuffer, no Accelerate](docs/ArrayBufferW/OFramework.heic)

*Scalar Swift, ArrayBuffer, no Apple Accelerate – ~11.16× faster than JS in my test*

### NOTE

The tests were run on a base iPhone 12 running iOS 16.1.  
JS baseline uses the Vercel AI SDK [`cosineSimilarity`](https://github.com/vercel/ai/blob/main/packages/ai/src/util/cosine-similarity.ts).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/DaveyEke/react-native-nitro-cos-sim.git
cd react-native-nitro-cos-sim

# Install dependencies
npm install

# Generate Nitro bindings
npx nitrogen

# Run the example app (iOS Release)
cd example
npm install
npx expo prebuild --clean
npx expo run:ios --configuration Release
```

If you haven’t set up Nitro in your project yet, follow the Nitro Modules docs first:  
https://github.com/mrousavy/nitro

---

## Usage

### 1. Fastest path: `Float64Array` + `ArrayBuffer`

```ts
import { NitroCosSim } from 'react-native-nitro-cos-sim';

// Using typed arrays + ArrayBuffer (zero-copy into Swift)
const vec1 = new Float64Array([1, 2, 3, 4]);
const vec2 = new Float64Array([5, 6, 7, 8]);

const similarity = await NitroCosSim.cosineSimilarity(
  vec1.buffer,
  vec2.buffer
);

console.log('cosine similarity (ArrayBuffer):', similarity);
```

### 2. Convenience: regular JS arrays (slower, due to marshalling)

```ts
import { NitroCosSim } from 'react-native-nitro-cos-sim';

const similarity2 = await NitroCosSim.cosineSimilarity(
  [1, 2, 3, 4],
  [5, 6, 7, 8]
);

console.log('cosine similarity (arrays):', similarity2);
```

> For small vectors or one‑off calls, JS arrays are fine.  
> For larger vectors / many calls (e.g. embeddings), prefer `Float64Array` + `.buffer`.

---

## Example App / Benchmark

The example app compares:

- **JS**: `cosineSimilarity` from the Vercel AI SDK on plain JS arrays
- **Native**: Swift scalar implementation, using Nitro `ArrayBuffer` + pointers  
  (no Apple Accelerate / vDSP)

```tsx
// example/App.tsx (excerpt)
const vectorSize = 1536;
const iterations = 1000;

// JS uses regular arrays (Vercel AI cosineSimilarity baseline)
const jsVectorPairs = Array.from({ length: iterations }, () => ({
  a: randVector(vectorSize),
  b: randVector(vectorSize),
}));

// Native uses Float64Array + ArrayBuffer,
// Swift side is a plain scalar loop (NO Apple Accelerate framework).
const nativeVectorPairs = Array.from({ length: iterations }, () => ({
  a: randTypedVector(vectorSize),
  b: randTypedVector(vectorSize),
}));
```

The UI prints something like:

```text
Results (NO Accelerate):
------------------------
JS (Vercel AI SDK):          XXX.XXms
Native (scalar, ArrayBuffer): YYY.YYms

Per operation:
  JS:     0.XXXXms
  Native: 0.YYYYms

Speedup (JS / Native): 11.16x
```

---

## What I actually learned doing this

I started with the classic “native should be faster than JS” intuition, but a few details made a huge difference.

### 1. Don’t fight Nitro’s rules for large data

At first I was pushing big **standard JS arrays** through the bridge and wondering why it felt slow. Nitro’s performance docs are very clear:

> For large numeric data, **never** use normal JS arrays – use `ArrayBuffer` / typed arrays so native can work on the raw memory.

I was going against Nitro’s rule and paying for heavy array marshalling.

### 2. ArrayBuffer + pointers is enough – even without Apple’s Accelerate

Once I calmed down and actually read the Nitro docs:

1. I swapped the JS arrays for `Float64Array` and passed `.buffer` into Nitro.
2. On the Swift side, I **removed** Apple’s Accelerate / vDSP and went back to a simple scalar implementation, but this time with `ArrayBuffer` and pointers.

Key bits on the Swift side:

- `MemoryLayout<Double>.stride` is used to figure out how many bytes a single `Double` takes:

  ```swift
  let countA = sizeA / MemoryLayout<Double>.stride
  ```

- `UnsafeRawPointer(dataA).assumingMemoryBound(to: Double.self)` tells Swift:

  > “Treat this raw buffer as if it’s an array of `Double`s.”

  That gives me a `UnsafePointer<Double>` so I can just read `ptrA[i]` as a `Double`:

  ```swift
  let ptrA = UnsafeRawPointer(dataA).assumingMemoryBound(to: Double.self)
  let ptrB = UnsafeRawPointer(dataB).assumingMemoryBound(to: Double.self)

  for i in 0..<n {
    let a = ptrA[i]
    let b = ptrB[i]
    dotProduct += a * b
    magA += a * a
    magB += b * b
  }
  ```

No vDSP, no special Apple framework — just raw pointers, a scalar loop, and **ArrayBuffer** so there’s **zero copying** between JS and Swift.

### 3. Release mode is non‑negotiable for performance

In **Debug**, Swift keeps a lot of checks and turns off many optimizations, so tight numeric loops look much slower than they really are.

As soon as I ran the same benchmark in **Release**:

```bash
cd example
npx expo prebuild --clean
npx expo run:ios --configuration Release
```

the scalar Swift + `ArrayBuffer` version became **insanely** fast and ended up around **11.16× faster** than the JS cosine similarity implementation from Vercel AI in my test.

So this is basically a re‑implementation of the JS cosine similarity in Swift, using Nitro Modules, `ArrayBuffer`, and plain pointers — and it’s ~11.16× faster, boosted by Nitro Modules 🔥

---

## Building the example

To run the example benchmark:

```bash
# Generate Nitro bindings using nitrogen (Nitro codegen)
npx nitrogen

# Build and run the Expo example (iOS Release)
cd example
npx expo prebuild --clean
npx expo run:ios --configuration Release
```

Run on a real device for realistic performance.

---

## API

### `NitroCosSim.cosineSimilarity(a, b)`

- **Parameters**
  - `a`: `number[]` or `ArrayBuffer` (prefer `Float64Array.buffer`)
  - `b`: `number[]` or `ArrayBuffer`
- **Returns**: `Promise<number>` – cosine similarity in `[-1, 1]`
- **Throws** (native side): vector length mismatch if `a` and `b` are not the same length.

---

## Credits

Built with [Nitro Modules](https://github.com/mrousavy/nitro) by [@mrousavy](https://twitter.com/mrousavy).

---

## License

MIT
