from re import search
import librosa #Audio processing library
from pyparsing import empty
import soundfile as sf #For creating output files
import sys

from bisect import bisect_left
from math import floor

#Produces a time-stretched set of output samples based on the input samples
#inData: available input samples
#inDataSampleOffset: the sample index of the first input item
#outDataSampleOffset: the sample index of the first output item to produce
#speedMult: ratio of output sample
def writeTimeStretchedFile(inputSamples, outFileName, speedMult):
    
    inputZeroCrossings = librosa.zero_crossings(inputSamples)
    #List of the first sample in each available input oscillation
    inputOscStarts = [0] + inputZeroCrossings[1::2]

    #The expected total samples in the output file
    targetOutputSampleCount = round(len(inputSamples) / float(speedMult))

    fout = sf(outFileName, 'x+', sf, 2)
    
    nextOscFirstOutputSampleIdx = 0
    remainingOscillations = True
    while(remainingOscillations):
        #Algorithm to produce a new output oscillation as a
        #timestretched approximation of the corresponding input oscillation
        nextOscFirstInputSampleIdx = nextOscFirstOutputSampleIdx * speedMult

        #Step 1: retrieve the first input oscillation containing samples that overlap
        firstOverlappingInputOscIdx = bisect_left(inputOscStarts, nextOscFirstInputSampleIdx)
        firstOverlappingInputOscFirstSampleIdx = inputOscStarts[firstOverlappingInputOscIdx]
        firstOverlappingInputOscLastSampleIdx = -1
        if(firstOverlappingInputOscIdx == len(inputOscStarts) - 1):
            #First overlapping sample goes to end of input file
            #This means once we have an output copy of it, we have converted the whole file
            firstOverlappingInputOscLastSampleIdx = len(inputSamples - 1)
            remainingOscillations = False
        else:
            firstOverlappingInputOscLastSampleIdx = inputOscStarts[firstOverlappingInputOscIdx + 1] - 1

        firstOverlappingInputOsc = inputSamples[firstOverlappingInputOscFirstSampleIdx:firstOverlappingInputOscLastSampleIdx]

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
        while(moreOverlappingOscillations):
            overlapStartIdx = max(inputOscStarts[currentInputOscIdx], nextOscFirstInputSampleIdx)
            overlapEndIdx = min(inputOscStarts[currentInputOscIdx + 1], nextOscLastInputSampleIdx) - 1
            overlappingSampleCount = overlapEndIdx - overlapStartIdx
            
            overlappingOsc = []
            if(currentInputOscIdx >= len(inputOscStarts) - 1):
                #Reached last input oscillation in file
                overlappingOsc = inputSamples[inputOscStarts[currentInputOscIdx] : len(inputSamples) - 1]
                moreOverlappingOscillations = False
            else:
                overlappingOsc = inputSamples[ inputOscStarts[currentInputOscIdx] : inputOscStarts[currentInputOscIdx + 1] ]
                if(inputOscStarts[currentInputOscIdx + 1] > overlapEndIdx):
                    #This sample extends past the overlap zone
                    moreOverlappingOscillations = False

            overlappingOscAmp = max(overlappingOsc) - min(overlappingOsc)
            weightedOverlappingInputAmpSum += (overlappingOscAmp * overlappingSampleCount)
            
            overlappingInputOscCount += 1
            currentInputOscIdx += 1

        weightedOverlappingInputAmpAvg = weightedOverlappingInputAmpSum / overlappingInputOscCount

        #Step 4: create the new output oscillation as the first overlapping input oscillation
        #scaled to the weighted average amplitude of overlapping input oscillations
        firstOverlappingInputOscAmp = max(firstOverlappingInputOsc) - min(firstOverlappingInputOsc)
        outSamples = firstOverlappingInputOsc * (weightedOverlappingInputAmpAvg / firstOverlappingInputOscAmp)

        #Step 5: write the new oscillation to the file
        fout.write(outSamples)

        #Update indices and continue
        nextOscFirstOutputSampleIdx += len(outSamples)

    fout.save()
    fout.close()


# Defining main function
def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: zerowave.py --in --out --speedmult\n")
        print("--in: input file name")
        print("--out: output file name")
        print("--speedmult: speed multiplier (0.1-10.0")
        print("Example (150% speed): zerowave.py --song.wav --song150.wav --1.5\n")

    # Load the input file
    inFileName = str(args[0])
    inSamples, sr = librosa.load(inFileName, sr=None)

    #Verify the input file and other CL args
    if(len(inSamples) == 0):
        print("Could not open " + inFileName + ".\n")
        print("Check the spelling and ensure it is a valid audio file")
    
    outFileName = str(args[1])

    speedMult = float(args[2])
    if(speedMult < 10 or speedMult > 1000):
        print("Speed must be between 10% and 100\%")

    #Arguments and input file are valid. Begin producing output file
    writeTimeStretchedFile(inSamples, outFileName, speedMult)



    
    


# Using the special variable 
# __name__
if __name__=="__main__":
    main()
