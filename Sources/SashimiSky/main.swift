import Foundation
import CoreImage
import Metal
import SkyMaskCore

do {
    let args=Array(CommandLine.arguments.dropFirst())
    if args == ["--version"] { print("sashimi-sky 1.0.0 (AGPL-3.0-or-later)"); exit(0) }
    guard args.count==6, args[0]=="--model",args[2]=="--input",args[4]=="--output" else {
        throw SkyError.invalidModel("Usage: sashimi-sky --model MODEL.mlmodel --input IMAGE.png --output MASK.png")
    }
    let modelURL=URL(fileURLWithPath:args[1]),inputURL=URL(fileURLWithPath:args[3]),outputURL=URL(fileURLWithPath:args[5])
    guard inputURL.standardizedFileURL != outputURL.standardizedFileURL,
          let device=MTLCreateSystemDefaultDevice(),let image=CIImage(contentsOf:inputURL,options:[.applyOrientationProperty:true]),
          image.extent.width>0,image.extent.height>0,image.extent.width*image.extent.height<=40_000_000 else {throw SkyError.invalidModel("Invalid input image or Metal unavailable.")}
    let context=CIContext(mtlDevice:device,options:[.workingFormat:CIFormat.RGBAf])
    let model=try SkyMaskModel(url:modelURL)
    let mask=try model.mask(for:image,context:context)
    guard let data=context.pngRepresentation(of:mask,format:.L16,colorSpace:CGColorSpaceCreateDeviceGray()) else {throw SkyError.render}
    try data.write(to:outputURL,options:.atomic)
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription+"\n").utf8));exit(1)
}
