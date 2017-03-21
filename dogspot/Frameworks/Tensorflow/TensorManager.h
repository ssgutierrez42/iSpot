//
//  TensorManager.h
//  dogspot
//
//  Created by Santiago Gutierrez on 3/1/17.
//  Copyright © 2017 Santiago Gutierrez. All rights reserved.
//

#ifndef TensorManager_h
#define TensorManager_h

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>

@protocol TensorDelegate <NSObject>

@required
- (void)receivedPredictions:(NSArray*)predictions;

@end

@interface TensorManager : NSObject

@property (nonatomic, weak) id<TensorDelegate> delegate;

+ (TensorManager*)sharedManager;
- (BOOL)validSession;
- (void)initSession;
- (void)clearLabels;
- (void)closeSession;
- (void)runCNNOnFrame:(CVPixelBufferRef)pixelBuffer;
- (NSString*)readImage: (CGImageRef) image;

@end

#endif /* TensorManager_h */
