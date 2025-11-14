import type { HybridObject } from 'react-native-nitro-modules'

export interface CosineSimilarity extends HybridObject<{ ios: 'swift', android: 'kotlin' }> {
  cosineSimilarity(vectorA: ArrayBuffer, vectorB: ArrayBuffer): number
}
