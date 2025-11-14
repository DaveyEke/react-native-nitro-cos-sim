import Foundation
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
        
        let n = countA
        guard n > 0 else {
            return 0.0
        }
        
        let ptrA = UnsafeRawPointer(dataA).assumingMemoryBound(to: Double.self)
        let ptrB = UnsafeRawPointer(dataB).assumingMemoryBound(to: Double.self)
        
        var dotProduct: Double = 0
        var magA: Double = 0
        var magB: Double = 0
        
        for i in 0..<n {
            let a = ptrA[i]
            let b = ptrB[i]
            dotProduct += a * b
            magA += a * a
            magB += b * b
        }
        
        let denom = sqrt(magA) * sqrt(magB)
        return denom == 0 ? 0 : dotProduct / denom
    }
}
