import Foundation
import CoreML
import CoreImage

/// Optional, locally supplied SkyRemoval-compatible Core ML model. Weights are not bundled.
public final class SkyMaskModel {
    private let model: MLModel
    public init(url: URL) throws {
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at:url)
        defer { if compiled != url {try? FileManager.default.removeItem(at:compiled)} }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndGPU
        model = try MLModel(contentsOf:compiled,configuration:configuration)
        guard model.modelDescription.inputDescriptionsByName["modelInput"]?.multiArrayConstraint?.shape == [1,3,384,384],
              model.modelDescription.outputDescriptionsByName["modelOutput"]?.multiArrayConstraint?.shape == [1,1,384,384] else {
            throw SkyError.invalidModel("Expected a SkyRemoval model with 384 × 384 RGB input.")
        }
    }
    public func mask(for image: CIImage, context: CIContext) throws -> CIImage {
        let size = 384, extent = image.extent
        guard let cg = context.createCGImage(image,from:extent,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)!) else {throw SkyError.render}
        // Work with encoded sRGB values as numeric data, matching the training input.
        let numeric = CIImage(cgImage:cg,options:[.colorSpace:NSNull()])
        let resized = numeric.applyingFilter("CILanczosScaleTransform",parameters:["inputScale":Double(size)/extent.height,"inputAspectRatio":extent.height/extent.width])
        var pixels = [Float](repeating:0,count:size*size*4)
        pixels.withUnsafeMutableBytes { context.render(resized,toBitmap:$0.baseAddress!,rowBytes:size*16,bounds:CGRect(x:0,y:0,width:size,height:size),format:.RGBAf,colorSpace:nil) }
        let input = try MLMultiArray(shape:[1,3,384,384],dataType:.float32)
        let pointer = input.dataPointer.bindMemory(to:Float.self,capacity:input.count)
        for channel in 0..<3 {for i in 0..<size*size {pointer[channel*size*size+i] = min(1,max(0,pixels[i*4+channel]))}}
        let prediction = try model.prediction(from:MLDictionaryFeatureProvider(dictionary:["modelInput":input]))
        guard let output = prediction.featureValue(for:"modelOutput")?.multiArrayValue, output.count == size*size else {throw SkyError.render}
        var values = [Float](repeating:0,count:output.count)
        for i in values.indices {values[i] = output[i].floatValue}
        let data = values.withUnsafeBytes {Data($0)}
        let small = CIImage(bitmapData:data,bytesPerRow:size*4,size:CGSize(width:size,height:size),format:.Rf,colorSpace:nil)
        let probability = small.applyingFilter("CILanczosScaleTransform",parameters:["inputScale":extent.height/Double(size),"inputAspectRatio":extent.width/extent.height]).cropped(to:extent)
        return try Self.refine(probability, guide:numeric, radius:20)
    }
    private static let kernels: [String:CIKernel] = {
        let functions = [
            "moments": "float b=guide.b; return float4(b,b*b,p.r,1.0);",
            "product": "return float4(float3(guide.b*p.r),1.0);",
            "coefficients": "float a=(p.r-guide.r*guide.b)/(max(0.0,guide.g-guide.r*guide.r)+0.01); return float4(a,guide.b-a*guide.r,0.0,1.0);",
            "guidedOutput": "return float4(float3(clamp(p.r*guide.b+p.g,0.0,1.0)),1.0);"
        ]
        return functions.reduce(into:[:]) { result,entry in
            let source = """
            #include <metal_stdlib>
            #include <CoreImage/CoreImage.h>
            using namespace metal;
            extern "C" [[ stitchable ]] float4 \(entry.key)(coreimage::sample_t guide, coreimage::sample_t p) {
                \(entry.value)
            }
            """
            if let kernel = try? CIKernel.kernels(withMetalString:source).first {result[entry.key]=kernel}
        }
    }()
    static func refine(_ probability:CIImage,guide:CIImage,radius:Double) throws -> CIImage {
        let extent=guide.extent
        func kernel(_ name:String,_ inputs:[CIImage]) throws -> CIImage {
            guard let k=kernels[name], let image=k.apply(extent:extent,roiCallback:{_,r in r},arguments:inputs) else {throw SkyError.render}
            return image
        }
        func mean(_ image:CIImage)->CIImage {image.clampedToExtent().applyingFilter("CIBoxBlur",parameters:["inputRadius":radius]).cropped(to:extent)}
        let moments=try kernel("moments",[guide,probability]), product=try kernel("product",[guide,probability])
        let coefficients=try kernel("coefficients",[mean(moments),mean(product)])
        return try kernel("guidedOutput",[guide,mean(coefficients)])
    }
}

public enum SkyError: LocalizedError {
    case invalidModel(String), render
    public var errorDescription:String? {
        switch self {case .invalidModel(let message):return message;case .render:return "Sky mask rendering failed."}
    }
}
