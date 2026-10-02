// Apple Vision text recognition helper for the OCR benchmark.
//
// Usage: recognize <image-path>
// Prints one JSON object to stdout. On failure prints {"error": "..."} and
// exits 1. Compiled by runners/apple_vision_runner.py with `swiftc -O`.
//
// API facts used (developer.apple.com/documentation/vision, read 2026-09-30):
//  - VNRecognizeTextRequest: recognitionLevel, usesLanguageCorrection,
//    recognitionLanguages ("Specify the languages as ISO language codes"),
//    automaticallyDetectsLanguage (macOS 13+), revision / supportedRevisions /
//    currentRevision (VNRequest).
//  - VNRecognizedTextObservation.topCandidates(_:) -> [VNRecognizedText].
//  - VNRecognizedText.boundingBox(for: Range<String.Index>) throws ->
//    VNRectangleObservation? ; with recognitionLevel .accurate "The bounding
//    box has word precision".
//  - VNDetectedObjectObservation.boundingBox: "normalizes the coordinates to
//    the dimensions of the processed image, with the origin at the lower-left
//    corner of the image" -> flipped here to a top-left origin.

import CoreGraphics
import Foundation
import ImageIO
import Vision

func fail(_ message: String) -> Never {
    let payload: [String: Any] = ["error": message]
    if let data = try? JSONSerialization.data(withJSONObject: payload),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
    exit(1)
}

/// Vision normalised rect (bottom-left origin) -> [x0, y0, x1, y1] top-left origin.
func topLeftBox(_ r: CGRect) -> [Double] {
    return [Double(r.minX), Double(1.0 - r.maxY), Double(r.maxX), Double(1.0 - r.minY)]
}

/// Quadrilateral corners (bottom-left origin) -> [[x, y]] top-left origin,
/// in TL, TR, BR, BL order.
func topLeftPoly(_ o: VNRectangleObservation) -> [[Double]] {
    let pts = [o.topLeft, o.topRight, o.bottomRight, o.bottomLeft]
    return pts.map { [Double($0.x), Double(1.0 - $0.y)] }
}

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
    return String(cString: buffer)
}

let args = CommandLine.arguments
guard args.count == 2 else { fail("usage: recognize <image-path>") }
let url = URL(fileURLWithPath: args[1])

guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fail("cannot decode image at \(args[1])")
}

// Apply the file's EXIF orientation so coordinates refer to the upright image
// (the same frame the Python side and the ground truth use).
var orientation = CGImagePropertyOrientation.up
if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
   let raw = props[kCGImagePropertyOrientation] as? UInt32,
   let parsed = CGImagePropertyOrientation(rawValue: raw) {
    orientation = parsed
}
let quarterTurns: [CGImagePropertyOrientation] = [.left, .leftMirrored, .right, .rightMirrored]
let rotated = quarterTurns.contains(orientation)
let width = rotated ? cgImage.height : cgImage.width
let height = rotated ? cgImage.width : cgImage.height

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]
if #available(macOS 13.0, *) {
    // Explicit language list above; do not let Vision switch models per image.
    request.automaticallyDetectsLanguage = false
}

let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
let started = Date()
do {
    try handler.perform([request])
} catch {
    fail("VNImageRequestHandler.perform failed: \(error)")
}
let inferenceSeconds = Date().timeIntervalSince(started)

var lines: [[String: Any]] = []
var words: [[String: Any]] = []
var wordRangeFailures = 0

for (lineIndex, observation) in (request.results ?? []).enumerated() {
    guard let candidate = observation.topCandidates(1).first else { continue }
    let text = candidate.string
    lines.append([
        "text": text,
        "confidence": Double(candidate.confidence),
        "bbox": topLeftBox(observation.boundingBox),
        "poly": topLeftPoly(observation),
    ])
    text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: .byWords) {
        substring, range, _, _ in
        guard let word = substring, !word.isEmpty else { return }
        do {
            if let box = try candidate.boundingBox(for: range) {
                words.append([
                    "text": word,
                    "bbox": topLeftBox(box.boundingBox),
                    "poly": topLeftPoly(box),
                    "line": lineIndex,
                    "confidence": Double(candidate.confidence),  // line-level
                ])
            } else {
                wordRangeFailures += 1
            }
        } catch {
            wordRangeFailures += 1
        }
    }
}

let osv = ProcessInfo.processInfo.operatingSystemVersion
var output: [String: Any] = [
    "lines": lines,
    "words": words,
    "word_range_failures": wordRangeFailures,
    "image_size": [width, height],
    "exif_orientation": Int(orientation.rawValue),
    "inference_seconds": inferenceSeconds,
    "request": [
        "class": "VNRecognizeTextRequest",
        "revision": request.revision,
        "current_revision": VNRecognizeTextRequest.currentRevision,
        "supported_revisions": Array(VNRecognizeTextRequest.supportedRevisions),
        "recognition_level": "accurate",
        "uses_language_correction": request.usesLanguageCorrection,
        "recognition_languages": request.recognitionLanguages,
        "word_segmentation": "String.enumerateSubstrings(.byWords)",
    ] as [String: Any],
    "os_version": "\(osv.majorVersion).\(osv.minorVersion).\(osv.patchVersion)",
    "os_version_string": ProcessInfo.processInfo.operatingSystemVersionString,
]
if let build = sysctlString("kern.osversion") { output["os_build"] = build }
if let model = sysctlString("hw.model") { output["hardware_model"] = model }
if let cpu = sysctlString("machdep.cpu.brand_string") { output["cpu"] = cpu }

do {
    let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
} catch {
    fail("JSON encoding failed: \(error)")
}
