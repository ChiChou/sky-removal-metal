// swift-tools-version: 5.10
import PackageDescription
let package = Package(name:"SashimiSky",platforms:[.macOS(.v14)],products:[.executable(name:"sashimi-sky",targets:["SashimiSky"])],targets:[
    .target(name:"SkyMaskCore"),
    .executableTarget(name:"SashimiSky",dependencies:["SkyMaskCore"]),
    .testTarget(name:"SkyMaskCoreTests",dependencies:["SkyMaskCore"])
])
