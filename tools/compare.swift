// Usage: swift run.swift model.mlmodel input.f32 output.f32 [cpu|gpu]
import Foundation
import CoreML
import CoreImage
let args = CommandLine.arguments
let url = URL(fileURLWithPath:args[1])
let compiled = try MLModel.compileModel(at:url)
defer {try? FileManager.default.removeItem(at:compiled)}
let config = MLModelConfiguration(); config.computeUnits = args.last == "cpu" ? .cpuOnly : .cpuAndGPU
let model = try MLModel(contentsOf:compiled,configuration:config)
let data: Data
if args[2].hasSuffix(".f32") {data = try Data(contentsOf:URL(fileURLWithPath:args[2]))}
else {
    let image=CIImage(contentsOf:URL(fileURLWithPath:args[2]))!,context=CIContext()
    let cg=context.createCGImage(image,from:image.extent,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)!)!
    let numeric=CIImage(cgImage:cg,options:[.colorSpace:NSNull()])
    let resized=numeric.applyingFilter("CILanczosScaleTransform",parameters:["inputScale":384/image.extent.height,"inputAspectRatio":image.extent.height/image.extent.width])
    var rgba=[Float](repeating:0,count:384*384*4),nchw=[Float](repeating:0,count:384*384*3)
    rgba.withUnsafeMutableBytes {context.render(resized,toBitmap:$0.baseAddress!,rowBytes:384*16,bounds:CGRect(x:0,y:0,width:384,height:384),format:.RGBAf,colorSpace:nil)}
    for c in 0..<3 {for i in 0..<384*384 {nchw[c*384*384+i]=min(1,max(0,rgba[4*i+c]))}}
    data=nchw.withUnsafeBytes {Data($0)}
    try data.write(to:URL(fileURLWithPath:args[3]+".input.f32"))
}
let input = try MLMultiArray(shape:[1,3,384,384],dataType:.float32)
precondition(data.count == input.count*4)
_ = data.withUnsafeBytes { memcpy(input.dataPointer,$0.baseAddress!,data.count) }
let provider = try MLDictionaryFeatureProvider(dictionary:["modelInput":input])
let start = Date()
let result = try model.prediction(from:provider)
let output = result.featureValue(for:"modelOutput")!.multiArrayValue!
var floats = [Float](repeating:0,count:output.count)
for i in floats.indices {floats[i] = output[i].floatValue}
try floats.withUnsafeBytes {try Data($0).write(to:URL(fileURLWithPath:args[3]))}
print("Core ML",config.computeUnits,"seconds",Date().timeIntervalSince(start),"shape",output.shape)
