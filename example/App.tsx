import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, ScrollView } from 'react-native';
import { NitroCosSim } from 'react-native-nitro-cos-sim';
import { cosineSimilarity } from 'ai';
import { useState } from 'react';

export default function App() {
  const [results, setResults] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);

  const randVector = (size: number): number[] =>
    Array.from({ length: size }, () => Math.random());

  const randTypedVector = (size: number): Float64Array => {
    const arr = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      arr[i] = Math.random();
    }
    return arr;
  };

  const runBenchmark = async () => {
    setIsRunning(true);
    setResults('Running benchmark (Swift scalar loop WITHOUT Accelerate)...');
    await new Promise(resolve => setTimeout(resolve, 100));

    const vectorSize = 1536;
    const iterations = 1000;

    // JS uses regular arrays (Vercel AI cosineSimilarity baseline)
    const jsVectorPairs = Array.from({ length: iterations }, () => ({
      a: randVector(vectorSize),
      b: randVector(vectorSize),
    }));

    // Native uses Float64Array + ArrayBuffer,
    // but the Swift side is a plain scalar loop (NO Apple Accelerate framework).
    const nativeVectorPairs = Array.from({ length: iterations }, () => ({
      a: randTypedVector(vectorSize),
      b: randTypedVector(vectorSize),
    }));

    // Warm up both implementations
    cosineSimilarity(jsVectorPairs[0].a, jsVectorPairs[0].b);
    NitroCosSim.cosineSimilarity(
      nativeVectorPairs[0].a.buffer,
      nativeVectorPairs[0].b.buffer
    );

    // JS baseline
    const jsStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      cosineSimilarity(jsVectorPairs[i].a, jsVectorPairs[i].b);
    }
    const jsTime = performance.now() - jsStart;

    // Native: Swift scalar implementation WITHOUT Apple's Accelerate,
    // but still using zero-copy ArrayBuffer on the RN side.
    const nativeStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      NitroCosSim.cosineSimilarity(
        nativeVectorPairs[i].a.buffer,
        nativeVectorPairs[i].b.buffer
      );
    }
    const nativeTime = performance.now() - nativeStart;

    const speedup = jsTime / nativeTime;

    const output = `
Results (NO Accelerate):
------------------------
JS (Vercel AI SDK):          ${jsTime.toFixed(2)}ms
Native (scalar, ArrayBuffer): ${nativeTime.toFixed(2)}ms

Per operation:
  JS:     ${(jsTime / iterations).toFixed(4)}ms
  Native: ${(nativeTime / iterations).toFixed(4)}ms

Speedup (JS / Native): ${speedup.toFixed(2)}x
    `.trim();

    setResults(output);
    setIsRunning(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Nitro Cosine Similarity</Text>
      <Text style={styles.subtitle}>
        Comparing JS vs Swift scalar loop (ArrayBuffer, zero-copy, WITHOUT Apple&apos;s Accelerate framework).
      </Text>
      <Button
        title={isRunning ? 'Running...' : 'Run Benchmark'}
        onPress={runBenchmark}
        disabled={isRunning}
      />
      <ScrollView style={styles.scrollView}>
        {results ? <Text style={styles.results}>{results}</Text> : (
          <Text style={styles.hint}>
            Run in a Release build for realistic performance:
            {'\n'}
            npx expo run:ios --configuration Release
          </Text>
        )}
      </ScrollView>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    marginTop: 50,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginBottom: 20,
    textAlign: 'center',
  },
  scrollView: {
    flex: 1,
    width: '100%',
    marginTop: 20,
  },
  results: {
    fontFamily: 'monospace',
    fontSize: 11,
    lineHeight: 16,
  },
  hint: {
    textAlign: 'center',
    color: '#999',
    marginTop: 40,
  },
});