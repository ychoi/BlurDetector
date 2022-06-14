import Accelerate
import simd
import CoreImage.CIFilterBuiltins
import UIKit

// Konstants
fileprivate struct K {
    // Accelerate supports lengths of c*2^n, c is 1, 3, 5, or 15, n>=3.
    static let logN: vDSP_Length = 10 // exponent of log2; N=1024.
    static let n = 1 << logN // must be a power of 2, b10000000000 = 1024
    static let cnt = n * n // pixel count in square nxn
}

public struct Spectrum {
    public let uiimg: UIImage?
    public let values: [Float]
    let width: Int
    let height: Int
    public let mean: Float
    public let stdDev: Float
    
    // Pixel format for converting amplitude spectrum (UInt8) to CGImage
    @available(iOS 13.0, *)
    static let grayPixelFormat = vImage_CGImageFormat(bitsPerComponent: 8,
                                                      bitsPerPixel: 8,
                                                      colorSpace: CGColorSpaceCreateDeviceGray(),
                                                      bitmapInfo: CGBitmapInfo(rawValue: 0))
    
    public func cgImage() -> CGImage? {
        let pixelCount = width * height
        var uIntPixels = [UInt8](repeating: 0, count: pixelCount)
        
        // Convert Float array to UInt8 array
        if #available(iOS 13.0, *) {
            vDSP.convertElements(of: values, to: &uIntPixels, rounding: .towardZero)
        } else {
            // Fallback on earlier versions
        }
        
        // Create a CGImage from pixel values
        let cgimage: CGImage? = uIntPixels.withUnsafeMutableBufferPointer { uIntPixelsPtr in
            let buffer = vImage_Buffer(data: uIntPixelsPtr.baseAddress!,
                                       height: vImagePixelCount(height),
                                       width: vImagePixelCount(width),
                                       rowBytes: width)
            
            if #available(iOS 13.0, *) {
                if let format = Self.grayPixelFormat {
                    return try? buffer.createCGImage(format: format)
                }
                else {return nil}
            } else {
                // Fallback on earlier versions
            }
            return nil
        }
        return cgimage
    }
    
    public func decibel() -> Double {
////        if spectrum.mean > 0.0 {
////            DispatchQueue.main.asyncAfter(deadline: .now(), execute: {
////                print("Mean: \(spectrum.mean)")})
////        }
//
//    if spectrum.values.count > 0 {
//        let decibels = spectrum.values
//        var ignoreCnt = 0
//        var sum = Double(0)
//        for decibel in decibels {
//            if decibel < 0 { ignoreCnt += 1; continue}
//            sum += Double(decibel)
//        }
//
//        let cnt = Float(decibels.count - ignoreCnt)
//        if cnt > Float(decibels.count) * 0.9 {
//            /// decibel is a unit used to measure the intensity of a sound or the power level of an electrical signal by comparing it with a given level on a logarithmic (laa·guh·Rith·mik) scale. A logarithmic scale is a way of displaying numerical data over a very wide range of values in a compact way.
//            DispatchQueue.main.asyncAfter(deadline: .now(), execute: {
//                print("Decibel: \(Float(sum) / cnt) dB")
//            })
//        }
//    }
        return 0
    }
}

@available(iOS 10.0, *)
@available(iOS 13.0, *)
public class BlurDetector {
    private let ciContext = CIContext(options: [.cacheIntermediates : false])
    private let fftSetup: FFTSetup
    public var imageLength: Int {return K.n}
    public let fft2d = vDSP.FFT2D(width: K.n, height: K.n, ofType: DSPSplitComplex.self)

    public init() {
        // FFT setup once in init, and re-use it.
        if let fftSetup = vDSP_create_fftsetup(vDSP_Length(K.logN), FFTRadix(kFFTRadix2)) {
            self.fftSetup = fftSetup
        } else {fatalError()}
    }; deinit {vDSP_destroy_fftsetup(fftSetup)}
}

// Experiments, threshold, 5e+14, 5e+8, 5e+7
@available(iOS 13.0, *)
public extension BlurDetector {
    
    func detect(image: UIImage, mask: UIImage, threshold: Float = 5e+8) -> [Spectrum] {

        assert(image.size == CGSize(width: K.n, height: K.n) &&
               mask.size == CGSize(width: K.n, height: K.n),
               "Image size must be \(K.n) x \(K.n)")

        let width = Int(image.size.width)
        let height = Int(image.size.height)
        let cnt = (width * height)/2 //half of image area pixel count
        
        var sourceImageReal = [Float](repeating: 0, count: cnt)
        var sourceImageImaginary = [Float](repeating: 0, count: cnt)
        
        var maskImageReal = [Float](repeating: 0, count: cnt)
        var maskImageImag = [Float](repeating: 0, count: cnt)
        
        let result: [Spectrum] = sourceImageReal.withUnsafeMutableBufferPointer { sr in
            sourceImageImaginary.withUnsafeMutableBufferPointer { si in
                maskImageReal.withUnsafeMutableBufferPointer { mr in
                    maskImageImag.withUnsafeMutableBufferPointer { mi in
                        
                        var sc = DSPSplitComplex(realp: sr.baseAddress!, imagp: si.baseAddress!)
                        imageToComplex(image, splitComplex: &sc)
                        
                        var mc = DSPSplitComplex(realp: mr.baseAddress!, imagp: mi.baseAddress!)
                        imageToComplex(mask, splitComplex: &mc)
                        
                        return fft_2D(sSplitComplex: &sc, mSplitComplex: &mc,
                                      size: image.size, threshold: threshold)
                    }
                }
            }
        }
        
        return result
    }
    
    func fft_2D(sSplitComplex: inout DSPSplitComplex,
                mSplitComplex: inout DSPSplitComplex,
                size: CGSize, threshold: Float) -> [Spectrum] {
                
        var spectrums = [Spectrum]()
        var decibels: [Float] = []
        var mean = Float.nan
        var stdDev = Float.nan

        let width = Int(size.width)
        let height = Int(size.height)
        let cnt = (width * height)/2 //pixel count half
        
        var sRealFrequency = [Float](repeating: 0, count: cnt)
        var sImagFrequency = [Float](repeating: 0, count: cnt)
        var sAmplitude = [Float](repeating: 0, count: cnt)

        var mRealFrequency = [Float](repeating: 0, count: cnt)
        var mImagFrequency = [Float](repeating: 0, count: cnt)
        var mAmplitude = [Float](repeating: 0, count: cnt) // maskImageAmplitude
        
        sRealFrequency.withUnsafeMutableBufferPointer { sr in
            sImagFrequency.withUnsafeMutableBufferPointer { si in
                mRealFrequency.withUnsafeMutableBufferPointer { mr in
                    mImagFrequency.withUnsafeMutableBufferPointer { mi in
                        
                        var sFrequency = DSPSplitComplex( realp: sr.baseAddress!, imagp: si.baseAddress!)
                        fft2d?.transform(input: sSplitComplex, output: &sFrequency, direction: .forward)
                        vDSP.squareMagnitudes(sFrequency, result: &sAmplitude)

                        var mFrequency = DSPSplitComplex( realp: mr.baseAddress!, imagp: mi.baseAddress!)
                        fft2d?.transform(input: mSplitComplex, output: &mFrequency, direction: .forward)
                        
                        vDSP.squareMagnitudes(mFrequency, result: &mAmplitude)
                        
                    }
                }
            }
        }
   
        // Zero the Peaks, all values in `mAmplitude` greater than `threshold` to -1, else +1...
        let outputConstant: Float = -1
        vDSP.threshold(mAmplitude,
                       to: powf(10, 5+10*threshold),
                       with: .signedConstant(outputConstant),
                       result: &mAmplitude)
        
        // all negative values in `maskImageAmplitude` to 0...
        vDSP.clip(mAmplitude, to: 0 ... 1, result: &mAmplitude)
        
        // multiply frequency domain pixels by values in `maskImageAmplitude`...
        mAmplitude[0] = 1

        var realSpatial = [Float](repeating: 0, count: cnt) // floatPixelsReal_spatial
        var imagSpatial = [Float](repeating: 0, count: cnt)

        let image: UIImage? = sRealFrequency.withUnsafeMutableBufferPointer { srcR in
            sImagFrequency.withUnsafeMutableBufferPointer { srcI in
                realSpatial.withUnsafeMutableBufferPointer { maskR in
                    imagSpatial.withUnsafeMutableBufferPointer { maskI in
                        
                        var sFrequency = DSPSplitComplex( realp: srcR.baseAddress!, imagp: srcI.baseAddress!)
                        
                        vDSP.multiply(sFrequency, by: mAmplitude, result: &sFrequency)
                        
                        // perform inverse FFT to create image from frequency domain data...
                        var backToSpatial = DSPSplitComplex(realp: maskR.baseAddress!,
                                                                  imagp: maskI.baseAddress!)
                        fft2d?.transform(input: sFrequency, output: &backToSpatial, direction: .inverse)
                                                

                        // Convert complex numbers to amplitude
                        vDSP.absolute(sFrequency, result: &sAmplitude)
                        // Convert amplitude spectrum to decibels, power spectrum [dB]
                        decibels = vDSP.amplitudeToDecibels(sAmplitude, zeroReference: 1)

                        // To visualize frequency domain for debug purpose only.
                        // OPT: Swap the data in the upper half and the lower half.
                        decibels.withUnsafeMutableBufferPointer { ptr in
                            let p1 = UnsafeMutablePointer(ptr.baseAddress!)
                            let p2 = p1.advanced(by: cnt/2)
                            vDSP_vswap(p1, 1, p2, 1, vDSP_Length(cnt/2))
                        }
                        // OPT: Swap the data in the left half and the right half.
                        decibels.withUnsafeMutableBufferPointer { ptr in
                            let p1 = UnsafeMutablePointer(ptr.baseAddress!)
                            let p2 = p1.advanced(by: width/4) // 256
                            vDSP_vswap(p1, 1, p2, 1, vDSP_Length(cnt))
                        }
                           
                                                
                        // Calculate standard deviation for blur score
                        vDSP_normalize(backToSpatial.realp, 1, nil, 1, &mean, &stdDev, vDSP_Length(cnt))
                        //print("Blur Score: \(mean), \(stdDev)")

                        return imageFromPixelSource(&backToSpatial, width: width, height: height,
                                                    bitmapInfo: CGBitmapInfo(rawValue: 0))
                    }
                }
            }
        }
        
        if let image = image {
            spectrums.append(Spectrum(uiimg: image, values: decibels, width: K.n/2, height: K.n, mean: mean, stdDev: stdDev))
                        
            spectrums.append(Spectrum(uiimg: image, values: sAmplitude, width: K.n/2, height: K.n, mean: mean, stdDev: stdDev))
            
            spectrums.append(Spectrum(uiimg: image, values: mAmplitude, width: K.n/2, height: K.n, mean: mean, stdDev: stdDev))
        }

        return spectrums
    }
    
    func imageToComplex(_ image: UIImage, splitComplex splitComplexOut: inout DSPSplitComplex) {
        guard let cgImage = image.cgImage else {fatalError("unable to generate cgimage")}
        
        let pixelCount = Int(image.size.width * image.size.height)
        let pixelData = cgImage.dataProvider?.data
        let pixelsArray = Array(UnsafeBufferPointer(start: CFDataGetBytePtr(pixelData),
                                                    count: pixelCount))
        
        let floatPixels = vDSP.integerToFloatingPoint(pixelsArray,
                                                      floatingPointType: Float.self)
        
        let interleavedPixels = stride(from: 1, to: floatPixels.count, by: 2).map {
            return DSPComplex(real: floatPixels[$0.advanced(by: -1)],
                              imag: floatPixels[$0])
        }
        
        vDSP.convert(interleavedComplexVector: interleavedPixels,
                     toSplitComplexVector: &splitComplexOut)
    }
    
    func imageFromPixelSource(_ pixelSource: inout DSPSplitComplex,
                                     width: Int, height: Int,
                                     denominator: Float = pow(2, 29),
                                     bitmapInfo: CGBitmapInfo) -> UIImage? {
        
        let pixelCount = width * height
        let cnt = vDSP_Length(pixelCount / 2)
        let stride = vDSP_Stride(1)
        
        // multiply all float values (1 / denominator) * 255
        let multiplier = [Float](repeating: Float((1 / denominator) * 255),
                                 count: pixelCount / 2)

        vDSP.multiply(pixelSource,
                      by: multiplier,
                      result: &pixelSource)
        
        // Clip values to 0...255
        var low: Float = 0
        var high: Float = 255

        vDSP_vclip(pixelSource.realp,
                   stride,
                   &low,
                   &high,
                   pixelSource.realp,
                   stride,
                   cnt)
        
        vDSP_vclip(pixelSource.imagp,
                   stride,
                   &low,
                   &high,
                   pixelSource.imagp,
                   stride, cnt)
        
        var uIntPixels = [UInt8](repeating: 0, count: pixelCount)

        let floatPixels = [Float](fromSplitComplex: pixelSource, scale: 1, count: pixelCount)
        
        vDSP.convertElements(of: floatPixels, to: &uIntPixels, rounding: .towardZero)
        
        
        let result: UIImage? = uIntPixels.withUnsafeMutableBufferPointer { uIntPixelsPtr in
            
            let buffer = vImage_Buffer(data: uIntPixelsPtr.baseAddress!,
                                       height: vImagePixelCount(height),
                                       width: vImagePixelCount(width),
                                       rowBytes: width)
            
            if let format = vImage_CGImageFormat(bitsPerComponent: 8,
                                                  bitsPerPixel: 8,
                                                  colorSpace: CGColorSpaceCreateDeviceGray(),
                                                  bitmapInfo: bitmapInfo),
                let cgImage = try? buffer.createCGImage(format: format) {
                
                return UIImage(cgImage: cgImage)
            }
            else {
                print("Unable to create `CGImage`.")
                return nil
            }
        }

        return result
    }
}

@available(iOS 13.0, *)
public extension BlurDetector {
    
    // Convert Spatial domain to Frequency domain; amplitude spectrum
    // Input: float array for resized, ex:256x256, of captured image.
    // Output: an array of Spectrum which contains AccelerateMutableBuffer and score
    func detect2(image: [Float], w: Int, h: Int, debug: Bool = true) -> [Spectrum] {
        var spectrums = [Spectrum]()
        
        // complex data size definition
        // Where to store float values for the amplitudes,
        // AccelerateMutableBuffer with Float elements
        var amplitude = [Float](repeating: 0, count: w*h)
        var decibels: [Float] = []
        var mean = Float.nan
        var stdDev = Float.nan
        
        image.withUnsafeBytes { imagePtr in
            _ = [Float](unsafeUninitializedCapacity: w*h) { r, _ in
                _ = [Float](unsafeUninitializedCapacity: w*h) { i, _ in
                    // Create a `DSPSplitComplex` to receive the FFT result.
                    var sc = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                    
                    // Generate a split complex vector from the image data
                    // Copies the contents of an interleaved complex vector C to a split complex vector Z; single precision.
                    // converts the real values to the even-odd split configuration
                    // convert real spatial-domain values, for example, pixel intensities, to complex values.
                    vDSP_ctoz([DSPComplex](imagePtr.bindMemory(to: DSPComplex.self)),
                              2, &sc, 1, vDSP_Length((K.n / 2 * K.n)))
                    
                    // Perform the Forward FFT.
                    vDSP_fft2d_zrip(fftSetup, &sc, 1, 0, K.logN, K.logN,
                                    FFTDirection(kFFTDirection_Forward))
                    
                    // zrip results are 2x the standard FFT and need to be scaled
                    // OPT: Scale it, for debug purpose
                    var scale: Float = Float(1.0/2.0)
                    vDSP_vsmul(sc.realp, 1, &scale, sc.realp, 1, vDSP_Length(K.n / 2 * K.n))
                    vDSP_vsmul(sc.imagp, 1, &scale, sc.imagp, 1, vDSP_Length(K.n / 2 * K.n))
                    
                    // Zero out areas
                    sc.imagp[0] = 0.0 // the nyquist value
                    let maskSize = 45 // TBD size & shape
                    let j = Int(bitPattern: imagePtr.baseAddress!) / w
                    if (j >= 0 && j < Int(maskSize/2)) ||
                       (j > Int(h-maskSize/2) && j < Int(h)){
                        sc.realp.assign(repeating: 0, count: maskSize/2)
                    }
                    
                    // Perform the Inverse FFT to get Spacial Domain back.
                    if debug {
                        vDSP_fft2d_zrip(fftSetup, &sc, 1, 0, K.logN, K.logN, FFTDirection(kFFTDirection_Inverse))
                    }
                    
                    // Convert complex numbers to amplitude
                    vDSP.absolute(sc, result: &amplitude)
                    
                    // Convert amplitude spectrum to decibels, power spectrum [dB]
                    decibels = vDSP.amplitudeToDecibels(amplitude, zeroReference: 1)

                    
                    // OPT: Swap the data in the upper half and the lower half.
                    // To visualize frequency domain for debug purpose only.
                    if !debug {
                        decibels.withUnsafeMutableBufferPointer { ptr in
                            let p1 = UnsafeMutablePointer(ptr.baseAddress!)
                            let p2 = p1.advanced(by: w*h / 2)
                            vDSP_vswap(p1, 1, p2, 1, vDSP_Length(w*h / 2))
                        }
                    }
                    
                    // Calculate standard deviation for blur score
                    vDSP_normalize(decibels, 1, nil, 1, &mean, &stdDev, vDSP_Length(decibels.count))
                }
            }
        }
        
        // The main output spectrum.
        spectrums.append(Spectrum(uiimg: nil, values: decibels, width: w, height: h, mean: mean, stdDev: stdDev))
        // OPT: additional spectrums.
        spectrums.append(Spectrum(uiimg: nil, values: amplitude, width: w, height: h, mean: mean, stdDev: stdDev))

        return spectrums
    }
    
}

@available(iOS 13.0, *)
public extension BlurDetector {
    func detect33(image: [Float], w: Int, h: Int, debug: Bool = true) -> [Spectrum] {
        var spectrums = [Spectrum]()
        
        // complex data size definition
        // Where to store float values for the amplitudes,
        // AccelerateMutableBuffer with Float elements
        var amplitude = [Float](repeating: 0, count: w*h)
        var decibels: [Float] = []
        var mean = Float.nan
        var stdDev = Float.nan
        
        image.withUnsafeBytes { imagePtr in
            _ = [Float](unsafeUninitializedCapacity: w*h) { r, _ in
                _ = [Float](unsafeUninitializedCapacity: w*h) { i, _ in
                    // Create a `DSPSplitComplex` to receive the FFT result.
                    var sc = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                    
                    // Generate a split complex vector from the image data
                    // Copies the contents of an interleaved complex vector C to a split complex vector Z; single precision.
                    // converts the real values to the even-odd split configuration
                    // convert real spatial-domain values, for example, pixel intensities, to complex values.
                    vDSP_ctoz([DSPComplex](imagePtr.bindMemory(to: DSPComplex.self)),
                              2, &sc, 1, vDSP_Length((K.n / 2 * K.n)))
                    
                    // Perform the Forward FFT.
                    vDSP_fft2d_zrip(fftSetup, &sc, 1, 0, K.logN, K.logN,
                                    FFTDirection(kFFTDirection_Forward))
                    
                    // zrip results are 2x the standard FFT and need to be scaled
                    // OPT: Scale it, for debug purpose
                    var scale: Float = Float(1.0/2.0)
                    vDSP_vsmul(sc.realp, 1, &scale, sc.realp, 1, vDSP_Length(K.n / 2 * K.n))
                    vDSP_vsmul(sc.imagp, 1, &scale, sc.imagp, 1, vDSP_Length(K.n / 2 * K.n))
                    
                    // Zero out areas
                    sc.imagp[0] = 0.0 // the nyquist value
                    let maskSize = 45 // TBD size & shape
                    let j = Int(bitPattern: imagePtr.baseAddress!) / w
                    if (j >= 0 && j < Int(maskSize/2)) ||
                       (j > Int(h-maskSize/2) && j < Int(h)){
                        sc.realp.assign(repeating: 0, count: maskSize/2)
                    }
                    
                    // Perform the Inverse FFT to get Spacial Domain back.
                    if debug {
                        vDSP_fft2d_zrip(fftSetup, &sc, 1, 0, K.logN, K.logN, FFTDirection(kFFTDirection_Inverse))
                    }
                    
                    // Convert complex numbers to amplitude
                    vDSP.absolute(sc, result: &amplitude)
                    
                    // Convert amplitude spectrum to decibels, power spectrum [dB]
                    decibels = vDSP.amplitudeToDecibels(amplitude, zeroReference: 1)

                    
                    // OPT: Swap the data in the upper half and the lower half.
                    // To visualize frequency domain for debug purpose only.
                    if !debug {
                        decibels.withUnsafeMutableBufferPointer { ptr in
                            let p1 = UnsafeMutablePointer(ptr.baseAddress!)
                            let p2 = p1.advanced(by: w*h / 2)
                            vDSP_vswap(p1, 1, p2, 1, vDSP_Length(w*h / 2))
                        }
                    }
                    
                    // Calculate standard deviation for blur score
                    vDSP_normalize(decibels, 1, nil, 1, &mean, &stdDev, vDSP_Length(decibels.count))
                }
            }
        }
        
        // The main output spectrum.
        spectrums.append(Spectrum(uiimg: nil, values: decibels, width: w, height: h, mean: mean, stdDev: stdDev))
        // OPT: additional spectrums.
        spectrums.append(Spectrum(uiimg: nil, values: amplitude, width: w, height: h, mean: mean, stdDev: stdDev))

        return spectrums
    }

}

@available(iOS 13.0, *)
public extension BlurDetector {
    func detect55(sourceImage: UIImage, threshold: Float = 5e+8) -> [Spectrum] {
        let width = Int(sourceImage.size.width)
        let height = Int(sourceImage.size.height)
        let cnt = (width * height)/2 //pixel count half

        //let complexW = width / 2 // half
        //let complexH = height // whole
        
        var sourceImageReal = [Float](repeating: 0, count: cnt)
        var sourceImageImaginary = [Float](repeating: 0, count: cnt)
        //let imgData = sourceImage.jpegData(compressionQuality: 1.0)!

        ///let out: UIImage? = imgData.withUnsafeBytes { pSig -> UIImage? in

            let result: [Spectrum] = sourceImageReal.withUnsafeMutableBufferPointer { sr in
                sourceImageImaginary.withUnsafeMutableBufferPointer { si in
                        
                var sc = DSPSplitComplex(realp: sr.baseAddress!, imagp: si.baseAddress!)
                    imageToComplex(sourceImage, splitComplex: &sc)
                    
//                    vDSP_ctoz( [DSPComplex](pSig.bindMemory(to: DSPComplex.self)),
//                                2, &sc, 1, vDSP_Length(width/2 * height/2) )
                    
                    let realDimension = width
                    // The binary logarithm of `max(rowCount, columnCount)`.
                     let countLog2n = vDSP_Length(log2(Float(realDimension)))
                     if let fft = vDSP_create_fftsetup(countLog2n, FFTRadix(kFFTRadix2)) {
                
                         let dimensionLog2n = vDSP_Length(log2(Float(realDimension)))
                         vDSP_fft2d_zrip(fft, &sc,
                                         1, 0,
                                         dimensionLog2n, dimensionLog2n,
                                         FFTDirection(kFFTDirection_Forward))
                         
                         vDSP_destroy_fftsetup(fft)
                     }

                    var mc = sc
                    return fft_2D(sSplitComplex: &sc, mSplitComplex: &mc,
                                  size: sourceImage.size, threshold: threshold)
                }
            }
            return result
        //}
        //return out
    }
}

@available(iOS 13.0, *)
public extension BlurDetector {
    //    func detect56 (img: UIImage) -> UIImage? {
    //        let imgW = img.size.width
    //        let imgH = img.size.height
    //        let realCnt = imgW * imgH
    //
    //        let complexW = imgW / 2 // half
    //        let complexH = imgH // whole
    //
    //        let complexCnt = complexW * complexH
    //        var complexReals = [Float](repeating:0.0, count:Int(realCnt))
    //        var complexImaginaries = [Float](repeating:0.0, count:Int(realCnt))
    //
    //        let imgData = img.jpegData(compressionQuality: 1.0)!
    //
    //        let result: UIImage? = imgData.withUnsafeBytes { pSig in
    //            // 10
    //            complexReals = [Float](unsafeUninitializedCapacity: Int(complexCnt)) {rBuf, riCnt in
    //
    //                complexImaginaries = [Float](unsafeUninitializedCapacity: Int(complexCnt)) {iBuf, iiCnt in
    //
    //                    // 1
    //                    var sc = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
    //
    //                    // 2
    //                    vDSP_ctoz( [DSPComplex](pSig.bindMemory(to: DSPComplex.self)),
    //                                2, &sc, 1, vDSP_Length(complexCnt) )
    //
    //                    // 3
    //                    let log2nCnt = vDSP_Length(log2(Float(imgW)))
    //
    //                    // 4
    //                    if let fft = vDSP_create_fftsetup(log2nCnt, FFTRadix(kFFTRadix2)) {
    //
    //                        vDSP_fft2d_zrip(fft, &sc, 1, 0, log2nCnt, log2nCnt, FFTDirection(KFFTDirection_Forward))
    //
    //                        vDSP_destory_fftsetup(fft)
    //                    }
    //
    //                    iiCnt = complexCnt
    //                }
    //                riCnt = Int(complexCnt)
    //            }
    //        }
    //        return result
    //    }
}

