//
//  TensorCameraViewController.swift
//  dogspot
//
//  Created by Santiago Gutierrez on 3/3/17.
//  Copyright © 2017 Santiago Gutierrez. All rights reserved.
//

import UIKit
import AVFoundation
import Accelerate
import CoreLocation

struct TensorPayload {
    var tensors: [BreedTensor]
    var image: UIImage?
    var croppedImage: UIImage?
    var location: CLLocationCoordinate2D?
    
    init() {
        self.tensors = [BreedTensor]()
        self.croppedImage = nil
        self.image = nil
        self.location = nil
    }
}

protocol TensorCameraDelegate {
    func received(tensorPayload: TensorPayload)
}

class TensorCameraViewController: UIViewController {
    
    @IBOutlet weak var cameraTopLabel: UILabel!
    
    @IBOutlet weak var selectionSquareHeight: NSLayoutConstraint!
    @IBOutlet weak var selectionSquare: SelectionSquare!
    
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var cameraButton: UIButton!
    @IBOutlet weak var cancelButton: UIButton!
    
    @IBOutlet weak var findButtonsView: UIView!
    
    fileprivate var cameraManager: CameraManager? = nil
    fileprivate var capturedImageView: UIImageView? = nil
    fileprivate var capturedPayload = TensorPayload()
    
    fileprivate var predictionLayers = [CATextLayer]()
    
    fileprivate let locationManager = CLLocationManager()
    
    var delegate: TensorCameraDelegate?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        locationManager.requestWhenInUseAuthorization()
        
        if CLLocationManager.locationServicesEnabled() {
            locationManager.delegate = self
            locationManager.desiredAccuracy = kCLLocationAccuracyBest
        }
        
        if !TensorManager.shared().validSession() {
            TensorManager.shared().initSession() //init TensorFlow
        }
        
        let videoSpec = VideoSpec(fps: 3, size: CGSize(width: 1280, height: 720))
        cameraManager = CameraManager(withSpec: videoSpec, andView: cameraView)
        cameraManager?.delegate = self
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        guard let cameraManager = cameraManager else {return}
        TensorManager.shared().delegate = self
        cameraManager.startCapture()
        
        selectionSquareHeight.constant = UIScreen.main.bounds.width
        navigationController?.setNavigationBarHidden(true, animated: true)
        locationManager.startUpdatingLocation()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        guard let cameraManager = cameraManager else {return}
        cameraManager.resizePreview()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        guard let cameraManager = cameraManager else {return}
        TensorManager.shared().clearLabels()
        TensorManager.shared().delegate = nil
        cameraManager.stopCapture()
        
        navigationController?.setNavigationBarHidden(false, animated: true)
        super.viewWillDisappear(animated)
        locationManager.stopUpdatingLocation()
    }
    
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    override var shouldAutorotate: Bool {
        return false
    }
    
    fileprivate func removePredictionLabels(){
        for label in predictionLayers {
            label.removeFromSuperlayer()
        }
        predictionLayers.removeAll()
    }
    
    fileprivate func updateLabels(with tensor: BreedTensor, usingColor color: UIColor = UIColor.black.withAlphaComponent(0.6)){
        guard let confidence = tensor.confidence else {
            return //we got no valid percentage from this label
        }
        
        let percentage = Int(round(confidence*100)) //round up, plz
        let label = String(format: "\t%d%% \t%@", percentage, tensor.label.capitalized)
        
        let labelHeight: CGFloat = 26.0
        let statusBarHeight = UIApplication.shared.statusBarFrame.height
        let labelY = cameraTopLabel.bounds.origin.y + cameraTopLabel.bounds.height + (labelHeight * CGFloat(predictionLayers.count))
        let frame = CGRect(x: 0.0, y: statusBarHeight + labelY, width: UIScreen.main.bounds.width, height: labelHeight)
        
        let textLayer = CATextLayer()
        textLayer.backgroundColor = color.cgColor
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.frame = frame
        textLayer.alignmentMode = kCAAlignmentLeft
        textLayer.isWrapped = true
        let font = CTFontCreateWithName("System" as CFString, 18.0, nil)
        textLayer.font = font
        textLayer.fontSize = 18.0
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.string = label
        
        self.view.layer.addSublayer(textLayer)
        predictionLayers.append(textLayer)
    }
    
    fileprivate func getTensorLabels(fromData predictionData: String) -> [BreedTensor] {
        var result = [BreedTensor]()
        
        let entries = predictionData.components(separatedBy: ",")
        
        for entry in entries {
            guard !entry.isEmpty else { continue } //we must not be empty
            
            let predictions = entry.components(separatedBy: "-")
            guard predictions.count > 1 else { continue } //we must have valid components
            
            if var tensor = Breed.getTensorLabel(fromData: predictions[1]),
                let probability = Double(predictions[0]) {
                tensor.confidence = probability
                result.append(tensor)
            }
        }
        
        return result
    }
    
    fileprivate func startCapture() {
        cameraManager?.startCapture()
        capturedPayload = TensorPayload()
        
        if !cameraButton.isEnabled {
            cameraButton.isEnabled = true
        }
    }
    
    fileprivate func updateFindButtons(hide: Bool){
        self.findButtonsView.isHidden = hide
        self.cameraButton.isHidden = !hide
        self.cancelButton.isHidden = !hide
    }
    
    @IBAction func retakePicturePressed(_ sender: UIButton) {
        guard let capturedImageView = capturedImageView,
            !findButtonsView.isHidden else {
            return
        }
        
        updateFindButtons(hide: true)
        
        capturedImageView.removeFromSuperview()
        self.startCapture()
    }
    
    @IBAction func savePressed(_ sender: UIButton) {
        guard !findButtonsView.isHidden else {
                return
        }
        
        if capturedPayload.location == nil {
            print("No location...TODO.")
            return
        }
        
        self.dismiss(animated: true, completion: nil)
        delegate?.received(tensorPayload: capturedPayload) //send callback after dismissal
    }
    
    @IBAction func cameraPressed(_ sender: UIButton) {
        sender.isEnabled = false //disable the button that sent this, one picture is enough.
        
        SwiftOverlays.showBlockingWaitOverlayWithText("Analyzing image")
        cameraManager?.captureStill()
    }
    
    @IBAction func cameraOff(_ sender: UIButton) {
        self.dismiss(animated: true, completion: nil)
    }

}

extension TensorCameraViewController: ImageBufferDelegate {
    //MARK: ImageBufferDelegate
    func beginningCapture() {
        let shutterView = UIView(frame: UIScreen.main.bounds)
        shutterView.backgroundColor = UIColor.white
        shutterView.alpha = 1
        
        UIApplication.shared.keyWindow?.addSubview(shutterView)
        UIView.animate(withDuration: 1, animations: {
            shutterView.alpha = 0.0
        }, completion: {(finished:Bool) in
            shutterView.removeFromSuperview()
        })
    }
    
    func received(_ buffer: CVPixelBuffer) {
        TensorManager.shared().runCNN(onFrame: buffer)
    }
    
    func captured(_ image: UIImage?){
        cameraManager?.stopCapture()
        
        let width = selectionSquare.bounds.width
        let height = selectionSquare.bounds.height
        let cropFrame = CGRect(x: 0.0, y: 0.0, width: width, height: height)
        
        guard let normalImage = image?.normalizedImage(),
            let cropData = normalImage.cropImage(toSquare: cropFrame) else {
                
                self.startCapture()
                SwiftOverlays.removeAllBlockingOverlays()
                return
        }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let strongSelf = self else { return }
            
            if let predictionData = TensorManager.shared().read(cropData.cgImage) { //this can block the main queue
                let tensorLabels = strongSelf.getTensorLabels(fromData: predictionData)
                
                DispatchQueue.main.async {
                    
                    if !tensorLabels.isEmpty {
                        
                        let imageView = UIImageView(image: image)
                        imageView.frame = UIScreen.main.bounds
                        strongSelf.cameraView.addSubview(imageView) //preview the exact image taken
                        
                        strongSelf.removePredictionLabels()
                        for tensor in tensorLabels {
                            strongSelf.updateLabels(with: tensor, usingColor: UIColor.themeRed.withAlphaComponent(0.8))
                        }
                        
                        strongSelf.updateFindButtons(hide: false)
                        strongSelf.capturedImageView = imageView
                        
                        strongSelf.capturedPayload.tensors = tensorLabels
                        strongSelf.capturedPayload.image = normalImage
                        strongSelf.capturedPayload.croppedImage = cropData.uiImage
                        
                    } else {
                        
                        print("Nothing was found.")
                        strongSelf.startCapture()
                        
                    }
                    
                }
                
            }
            
            DispatchQueue.main.async {
                SwiftOverlays.removeAllBlockingOverlays()
            }
        }
        
    }
}

extension TensorCameraViewController: TensorDelegate {
    
    //MARK: TensorDelegate
    func receivedPredictions(_ predictions: [Any]!) {
        guard let capturing = cameraManager?.isCapturing(), capturing == true else {
            return //only process prediction if we're capturing. Else, stop.
        }
        
        self.removePredictionLabels()
        
        for prediction in predictions {
            guard let prediction = prediction as? NSDictionary,
                let rawLabel = prediction["label"] as? String,
                let rawValue = prediction["value"] as? Double,
                var tensorLabel = Breed.getTensorLabel(fromData: rawLabel) else {
                    continue
            }
            
            tensorLabel.confidence = rawValue
            
            self.updateLabels(with: tensorLabel)
            
            if predictionLayers.count > 3 {
                break //don't keep adding stuff
            }
        }
    }
}

extension TensorCameraViewController: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        let location = manager.location!.coordinate
        self.capturedPayload.location = location
    }
}


