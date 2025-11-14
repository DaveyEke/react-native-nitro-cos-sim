import Foundation
import Accelerate
import NitroModules

enum CosineSimilarityError: Error {
    case vectorLengthMismatch
}

class HybridCosineSimilarity: HybridCosineSimilaritySpec {
    func cosineSimilarity(vectorA: NitroModules.ArrayBuffer, vectorB: NitroModules.ArrayBuffer) throws -> Double {
        let dataA = vectorA.data
        let dataB = vectorB.data
        
        let sizeA = vectorA.size
        let sizeB = vectorB.size
        
        let countA = sizeA / MemoryLayout<Double>.stride
        let countB = sizeB / MemoryLayout<Double>.stride
        
        guard countA == countB else {
            throw CosineSimilarityError.vectorLengthMismatch
        }
        
        let n = vDSP_Length(countA)
        
        guard n > 0 else {
            return 0.0
        }
        
        // Cast to Double pointers (zero-copy!)
        let ptrA = UnsafeRawPointer(dataA).assumingMemoryBound(to: Double.self)
        let ptrB = UnsafeRawPointer(dataB).assumingMemoryBound(to: Double.self)
        
        var dotProduct: Double = 0
        var magA: Double = 0
        var magB: Double = 0
        
        // vDSP SIMD operations with direct pointer access
        vDSP_dotprD(ptrA, 1, ptrB, 1, &dotProduct, n)
        vDSP_svesqD(ptrA, 1, &magA, n)
        vDSP_svesqD(ptrB, 1, &magB, n)
        
        let denom = sqrt(magA) * sqrt(magB)
        
        return denom == 0 ? 0 : dotProduct / denom
    }
}
