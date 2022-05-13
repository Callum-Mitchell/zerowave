from re import search
from tkinter import N
import librosa
from pyparsing import empty
import soundfile as sf #For creating output files
import sys

import numpy as np
from bisect import bisect_left
from math import floor

def transpose2DList(list2D):
    flatList = [cell for row in list2D for cell in row]
    print(flatList[:50])
    print(len(list2D[0]))
    print(len(list2D))
    return [flatList[e::len(list2D[0])] for e in range(len(list2D[0]))]


#Produces a time-stretched set of output samples based on the input samples
#inData: available input samples
#inDataSampleOffset: the sample index of the first input item
#outDataSampleOffset: the sample index of the first output item to produce
#speedMult: ratio of output sample
def getTimeStretchedSamples(inputSamples, speedMult, numChannels):

    IN_OSC_PER_WINDOW = 15
    outSamples = []
    targetOutSamplesPerChannel = round(len(inputSamples) / speedMult)
    for ch in range(0, numChannels):
        inputSamplesCurCh = inputSamples[:,ch]
        outSamplesCurCh = []
        #print(len(inputSamplesCurCh))
        inputZeroCrossingFlags = librosa.zero_crossings(inputSamplesCurCh)

        inputZeroCrossings = []
        for i in range(0, len(inputZeroCrossingFlags)):
            if(inputZeroCrossingFlags[i]):
                inputZeroCrossings.append(i)

        #print(inputSamplesCurCh[:100])
        #List of the first sample in each available input oscillation
        inputOscStarts = inputZeroCrossings[0::(2*IN_OSC_PER_WINDOW)]
        #print(inputOscStarts[:40])
        nextOscFirstOutputSampleIdx = 0
        remainingOscillations = True
        while remainingOscillations:
            #Algorithm to produce a new output oscillation as a
            #timestretched approximation of the corresponding input oscillation
            nextOscFirstInputSampleIdx = round(nextOscFirstOutputSampleIdx * speedMult)
            #print(nextOscFirstOutputSampleIdx)
            #print(nextOscFirstInputSampleIdx)
            #Step 1: retrieve the first input oscillation containing samples that overlap
            firstOverlappingInputOscIdx = bisect_left(inputOscStarts, nextOscFirstInputSampleIdx)
            #print(firstOverlappingInputOscIdx)
            if firstOverlappingInputOscIdx == len(inputOscStarts):
                #No overlapping input oscillation
                print('End of input stream reached')
                remainingOscillations = False
                break

            firstOverlappingInputOscFirstSampleIdx = inputOscStarts[firstOverlappingInputOscIdx]
            firstOverlappingInputOscLastSampleIdx = -1
            if firstOverlappingInputOscIdx >= len(inputOscStarts) - IN_OSC_PER_WINDOW:
                #First overlapping sample goes to end of input file
                #This means once we have an output copy of it, we have converted the whole file
                firstOverlappingInputOscLastSampleIdx = len(inputSamples) - 1
                remainingOscillations = False
            else:
                firstOverlappingInputOscLastSampleIdx = inputOscStarts[firstOverlappingInputOscIdx + IN_OSC_PER_WINDOW]

            #print(firstOverlappingInputOscFirstSampleIdx)
            #print(firstOverlappingInputOscLastSampleIdx)
            firstOverlappingInputOsc = inputSamplesCurCh[firstOverlappingInputOscFirstSampleIdx:firstOverlappingInputOscLastSampleIdx]

            #print(firstOverlappingInputOsc)
            #Step 2: determine the new output oscillation's length and in/out end sample indices
            nextOscSampleCount = len(firstOverlappingInputOsc)
            nextOscLastOutputSampleIdx = nextOscFirstOutputSampleIdx + nextOscSampleCount
            nextOscLastInputSampleIdx = nextOscLastOutputSampleIdx * speedMult

            #Step 3: determine the time-weighted average amplitude of input oscillations
            #which overlap with the output oscillation
            weightedOverlappingInputAmpSum = 0.0
            currentInputOscIdx = firstOverlappingInputOscIdx
            overlappingInputOscCount = 0
            moreOverlappingOscillations = True
            while moreOverlappingOscillations:

                #Start and end sample index for the overlapping input oscillation
                overlappingOscStartIdx = inputOscStarts[currentInputOscIdx]
                overlappingOscEndIdx = 0
                if currentInputOscIdx >= len(inputOscStarts) - 1:
                    overlappingOscEndIdx = len(inputSamples) - 1
                else:
                    overlappingOscEndIdx = inputOscStarts[currentInputOscIdx + 1]

                #Start and end sample index for just the overlapping section of the input oscillation
                overlapStartIdx = max(overlappingOscStartIdx, nextOscFirstInputSampleIdx)
                overlapEndIdx = min(overlappingOscEndIdx, nextOscLastInputSampleIdx) - 1
                overlappingSampleCount = overlapEndIdx - overlapStartIdx
            
                overlappingOsc = []
                if(currentInputOscIdx >= len(inputOscStarts) - 1):
                    #Reached last input oscillation in file
                    overlappingOsc = inputSamplesCurCh[inputOscStarts[currentInputOscIdx] : len(inputSamples) - 1 ]
                    moreOverlappingOscillations = False
                else:
                    overlappingOsc = inputSamplesCurCh[ inputOscStarts[currentInputOscIdx] : inputOscStarts[currentInputOscIdx + 1] ]
                    if(inputOscStarts[currentInputOscIdx + 1] > overlapEndIdx):
                        #This sample extends past the overlap zone
                        moreOverlappingOscillations = False

                overlappingOscAmp = max(overlappingOsc) - min(overlappingOsc)
                weightedOverlappingInputAmpSum += (overlappingOscAmp * overlappingSampleCount)
            
                overlappingInputOscCount += 1
                currentInputOscIdx += 1

            weightedOverlappingInputAmpAvg = weightedOverlappingInputAmpSum / (overlappingInputOscCount * len(firstOverlappingInputOsc))

            #Step 4: create the new output oscillation as the first overlapping input oscillation
            #scaled to the weighted average amplitude of overlapping input oscillations
            firstOverlappingInputOscAmp = max(firstOverlappingInputOsc) - min(firstOverlappingInputOsc)
            newOutOsc = firstOverlappingInputOsc # * (weightedOverlappingInputAmpAvg / firstOverlappingInputOscAmp)
            outSamplesCurCh.extend(newOutOsc)
            #print(len(newOutOsc))
            #print(len(outSamplesCurCh))
            #print(newOutOsc)
            #Update indices and continue
            nextOscFirstOutputSampleIdx = len(outSamplesCurCh)
        
        print(outSamplesCurCh[:50])
        #If output samples set is too short, append with more zeroes
        while(len(outSamplesCurCh) < targetOutSamplesPerChannel):
            outSamplesCurCh.append(0.0)
        outSamples.append(outSamplesCurCh[:targetOutSamplesPerChannel])

    return np.array(outSamples).transpose()


# Defining main function
def main():
    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: zerowave.py --in --out --speedmult --channels\n")
        print("--in: input file name")
        print("--out: output file name")
        print("--speedmult: speed multiplier (0.1-10.0)")
        print("Example (150% speed): zerowave.py stereosong.wav stereosong150.wav 1.5\n")

    # Load the input file
    inFileName = str(args[0])
    fin = sf.SoundFile(inFileName, 'r')
    inSamples = fin.read()
    sr = fin.samplerate
    fmt = fin.format
    sbtp = fin.subtype
    channels = fin.channels
    #Verify the input file and other CL args
    if len(inSamples) == 0:
        print("Could not open " + inFileName + ".\n")
        print("Check the spelling and ensure it is a valid audio file")
    
    outFileName = str(args[1])

    speedMult = float(args[2])
    if speedMult < 0.1 or speedMult > 10:
        print("Speed must be between 0.1 and 10")

    #Arguments and input file are valid. Begin producing output file
    outSamples = getTimeStretchedSamples(inSamples, speedMult, channels)

    print(inSamples[:30])
    print(outSamples[:30])
    print(len(inSamples))
    print(len(outSamples))

    #outSamplesMono = librosa.to_mono(outSamples.transpose)
    #print(len(outSamplesMono))
    print('Writing to file...')
    #Write the new oscillation to the file
    sf.write(file=outFileName, data=outSamples, samplerate=sr, format='wav')
    print(outFileName + ' created successfully')



# Using the special variable 
# __name__
if __name__=="__main__":
    main()
