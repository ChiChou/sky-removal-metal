import XCTest
import CoreImage
import Metal
@testable import SkyMaskCore

final class SkyMaskTests:XCTestCase {
    func testMetalGuidedFilterPreservesConstantMaskAndBoundary() throws {
        guard let device=MTLCreateSystemDefaultDevice() else {throw XCTSkip("Metal unavailable")}
        let context=CIContext(mtlDevice:device,options:[.workingFormat:CIFormat.RGBAf]), rect=CGRect(x:0,y:0,width:120,height:80)
        let guide=CIImage(color:CIColor(red:0.2,green:0.5,blue:0.8)).cropped(to:rect)
        let p=CIImage(color:CIColor(red:0.4,green:0.4,blue:0.4)).cropped(to:rect)
        let result=try SkyMaskModel.refine(p,guide:guide,radius:20)
        var values=[Float](repeating:0,count:120*80*4)
        values.withUnsafeMutableBytes {context.render(result,toBitmap:$0.baseAddress!,rowBytes:120*16,bounds:rect,format:.RGBAf,colorSpace:nil)}
        // Numeric CI constants are converted to the working space consistently on input/output.
        var expected=[Float](repeating:0,count:4)
        expected.withUnsafeMutableBytes {context.render(p,toBitmap:$0.baseAddress!,rowBytes:16,bounds:CGRect(x:0,y:0,width:1,height:1),format:.RGBAf,colorSpace:nil)}
        for i in stride(from:0,to:values.count,by:4000) {XCTAssertEqual(values[i],expected[0],accuracy:0.001);XCTAssertEqual(values[i+3],1,accuracy:0.001)}
    }
}
