import Foundation
import ImageIO
import Vision

let maximumInputBytes: UInt64 = 64 * 1024 * 1024

struct Token: Codable {
    let text: String
    let confidence: Float
    let bbox: [Double]
}

enum HelperError: Error {
    case invalidImage
}

let recognitionLevel: VNRequestTextRecognitionLevel
switch CommandLine.arguments.dropFirst().first {
case "fast":
    recognitionLevel = .fast
case "accurate", nil:
    recognitionLevel = .accurate
default:
    exit(4)
}

func readExactly(_ count: Int, from handle: FileHandle) -> Data? {
    var result = Data()
    while result.count < count {
        do {
            guard
                let chunk = try handle.read(upToCount: count - result.count),
                !chunk.isEmpty
            else {
                return nil
            }
            result.append(chunk)
        } catch {
            return nil
        }
    }
    return result
}

func decodeLength(_ data: Data) -> UInt64 {
    data.reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
}

func encodeLength(_ value: UInt64) -> Data {
    Data((0..<8).map { offset in
        UInt8((value >> UInt64((7 - offset) * 8)) & 0xff)
    })
}

func recognize(
    _ input: Data,
    recognitionLevel: VNRequestTextRecognitionLevel
) throws -> Data {
    try autoreleasepool {
        guard
            let source = CGImageSourceCreateWithData(input as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw HelperError.invalidImage
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = recognitionLevel
        request.usesLanguageCorrection = false
        request.recognitionLanguages = ["en-US"]
        try VNImageRequestHandler(cgImage: image).perform([request])

        let width = Double(image.width)
        let height = Double(image.height)
        let tokens = (request.results ?? []).compactMap { observation -> Token? in
            guard let candidate = observation.topCandidates(1).first else {
                return nil
            }
            let box = observation.boundingBox
            return Token(
                text: candidate.string,
                confidence: candidate.confidence,
                bbox: [
                    box.minX * width,
                    (1.0 - box.maxY) * height,
                    box.maxX * width,
                    (1.0 - box.minY) * height,
                ]
            )
        }
        return try JSONEncoder().encode(tokens)
    }
}

let inputHandle = FileHandle.standardInput
let outputHandle = FileHandle.standardOutput

while let header = readExactly(8, from: inputHandle) {
    let inputLength = decodeLength(header)
    guard
        inputLength > 0,
        inputLength <= maximumInputBytes,
        inputLength <= UInt64(Int.max),
        let input = readExactly(Int(inputLength), from: inputHandle)
    else {
        exit(2)
    }

    do {
        let output = try recognize(input, recognitionLevel: recognitionLevel)
        outputHandle.write(encodeLength(UInt64(output.count)))
        outputHandle.write(output)
    } catch {
        exit(3)
    }
}
