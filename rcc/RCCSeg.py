# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# March 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
import SimpleITK as sitk
from roc import ComputeROC
from rcc_common import LoadImage, LoadMask, LoadMaskNoRelabel, SaveImage, ComputeLabelWeights, ShowWarnings, ExtractTumorDetections, CleanUpMask
#from UNet import UNet
from Net import Net
#from diceloss import DiceLoss, make_one_hot
from Deterministic import NotDeterministic
#from TumorDSC import CalculateTumorDSC
import RCCMetrics

def extract_metrics(metricsByPatient, key, defaultVal=-1):
    mean = [defaultVal]*4
    std = [defaultVal]*4
    median = [defaultVal]*4

    for label in (1,2,3):
        if not any(key in metricsByLabel[label] for metricsByLabel in metricsByPatient.values()):
            continue

        if key == "mask_dsc":
            # NOTE: mask_dsc is just a single value
            valuesForLabel = np.concatenate([ [ metricsByLabel[label][key] ] for metricsByLabel in metricsByPatient.values() if key in metricsByLabel[label] ])
        else:
            valuesForLabel = np.concatenate([ metricsByLabel[label][key] for metricsByLabel in metricsByPatient.values() if key in metricsByLabel[label] ])

        mean[label] = valuesForLabel.mean()
        std[label] = valuesForLabel.std()
        median[label] = np.median(valuesForLabel)

    return mean, std, median

def extract_counts(metricsByPatient):
    totalCounts = [0]*4
    missedCounts = [0]*4
    extraCounts = [0]*4

    for label in (1,2,3):
        counts = np.array([ [ metricsByLabel[label]["gt_object_count"], metricsByLabel[label]["gt_missed_count"], metricsByLabel[label]["seg_extra_count"] ] for metricsByLabel in metricsByPatient.values() ])
        totalCounts[label], missedCounts[label], extraCounts[label] = counts.sum(axis=0)

    return totalCounts, missedCounts, extraCounts

def calculate_roc(metricsByPatient):
    if not any("tumor_detections" in metricsByLabel[3] for metricsByLabel in metricsByPatient.values()):
        return

    scoreAndLabels = np.concatenate([ metricsByLabel[3]["tumor_detections"] for metricsByLabel in metricsByPatient.values() if "tumor_detections" in metricsByLabel[3] and len(metricsByLabel[3]["tumor_detections"]) > 0 ], axis=0)
    scores = torch.from_numpy(scoreAndLabels[:, 0])
    labels = torch.from_numpy(scoreAndLabels[:, 1])

    roc = ComputeROC(scores, labels)

    return roc

class RCCSeg:
    def __init__(self, numClasses=4,use_expand=True):
        self.device = "cpu"
        self.multipleOf = [8, 8]
        self.numClasses=numClasses
        #self.net = UNet(in_channels=4,out_channels=self.numClasses)
        self.net = Net(in_channels=4,out_channels=self.numClasses, use_expand = use_expand)
        self.dataRoot = None
        #self.saveSteps = 1
        #self.valSteps = 5*self.saveSteps
        self.valSteps = 1
        self.dilateUnknown = False
        self.use_expand = use_expand
        self.epoch = 0

    def _get_roi_1d(self, size, multipleOf):
        remainder = (size % multipleOf)

        begin = int(remainder/2)
        end = begin + size - remainder

        return begin, end

    def _resize_image(self, npImg):
        beginX, endX = self._get_roi_1d(npImg.shape[-1], self.multipleOf[-1])
        beginY, endY = self._get_roi_1d(npImg.shape[-2], self.multipleOf[-2])

        return npImg[..., beginY:endY, beginX:endX].copy()

    def _pad_image(self, npImg, shape):
        beginX, endX = self._get_roi_1d(shape[-1], self.multipleOf[-1])
        beginY, endY = self._get_roi_1d(shape[-2], self.multipleOf[-2])

        npImgOutput = np.zeros(shape, npImg.dtype)
        npImgOutput[..., beginY:endY, beginX:endX] = npImg[...]

        return npImgOutput

    def SetDevice(self, device):
        self.device = device
        self.net = self.net.to(device)

    def GetDevice(self):
        return self.device

    def SetDataRoot(self, dataRoot):
        self.dataRoot = dataRoot

    def GetDataRoot(self):
        return self.dataRoot

    def SaveModel(self, fileName):
        torch.save({"epoch": self.epoch, "params": self.net.state_dict()}, fileName)

    def LoadModel(self, fileName):
        params = torch.load(fileName, map_location=self.GetDevice(), weights_only=True)

        if "epoch" in params:
            self.epoch = params["epoch"]
            self.net.load_state_dict(params["params"])
        else:
            # NOTE: Only style model
            self.use_expand = False
            self.net.load_state_dict(params)
            self.net.expand = None

    def LeafMaps(self,patientId):
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "image.nii.gz")
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "normalized_aligned.nii.gz")
        #volume2Path = os.path.join(self.dataRoot, "Images", patientId, "normalized2_aligned.nii.gz")

        sitkVolumes = [None]*4
        npVolumes = [None]*len(sitkVolumes)

        for i in range(len(npVolumes)):
            maskFileName = "normalized_aligned.nii.gz" if i == 0 else f"normalized{i+1}_aligned.nii.gz"
            volumePath = os.path.join(self.dataRoot, "Images", patientId, maskFileName)

            sitkVolumes[i] = LoadImage(volumePath)

            if sitkVolumes[i] is None:
                return None

            npVolumes[i] = sitk.GetArrayViewFromImage(sitkVolumes[i])

            if npVolumes[0].shape != npVolumes[i].shape:
                raise RuntimeError("Error: Dimension mismatch between volumes ({npVolumes[0].shape} != {i}: {npVolumes[i].shape}).")


        halfXDim = npVolumes[0].shape[2]//2

        npVolumesRight = [ npVolume[:, None, ..., :halfXDim] for npVolume in npVolumes ]
        npVolumesLeft = [ npVolume[:, None, ..., halfXDim:] for npVolume in npVolumes ]

        batchRight = self._resize_image(np.concatenate(tuple(npVolumesRight), axis=1)).astype(np.float32)
        batchLeft = self._resize_image(np.concatenate(tuple(npVolumesLeft), axis=1)).astype(np.float32)

        batchLeft = torch.from_numpy(batchLeft).to(self.GetDevice())
        batchRight = torch.from_numpy(batchRight).to(self.GetDevice())

        leafMaps = []

        with torch.no_grad():
            self.net.eval()

            outputLeft = self.net.leafmap(batchLeft)
            outputRight = self.net.leafmap(batchRight)

            self.net.train()

        outputRight = self._pad_image(outputRight.cpu().numpy(), [npVolumesRight[0].shape[0], outputRight.shape[1]] + list(npVolumesRight[0].shape[2:]))
        outputLeft = self._pad_image(outputLeft.cpu().numpy(), [npVolumesLeft[0].shape[0], outputLeft.shape[1]] + list(npVolumesLeft[0].shape[2:]))

        output = np.concatenate((outputRight, outputLeft), axis=-1)

        for t in range(output.shape[1]):
            leafMap = sitk.GetImageFromArray(output[:, t, ...].astype(np.int32))

            leafMap.SetSpacing(sitkVolumes[0].GetSpacing())
            leafMap.SetDirection(sitkVolumes[0].GetDirection())
            leafMap.SetOrigin(sitkVolumes[0].GetOrigin())

            leafMaps.append(leafMap)

        return leafMaps

    def RunOne(self,patientId):
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "image.nii.gz")
        #volumePath = os.path.join(self.dataRoot, "Images", patientId, "normalized_aligned.nii.gz")
        #volume2Path = os.path.join(self.dataRoot, "Images", patientId, "normalized2_aligned.nii.gz")

        sitkVolumes = [None]*4
        npVolumes = [None]*len(sitkVolumes)

        for i in range(len(npVolumes)):
            maskFileName = "normalized_aligned.nii.gz" if i == 0 else f"normalized{i+1}_aligned.nii.gz"
            volumePath = os.path.join(self.dataRoot, "Images", patientId, maskFileName)

            sitkVolumes[i] = LoadImage(volumePath)

            if sitkVolumes[i] is None:
                return None

            npVolumes[i] = sitk.GetArrayViewFromImage(sitkVolumes[i])

            if npVolumes[0].shape != npVolumes[i].shape:
                raise RuntimeError("Error: Dimension mismatch between volumes ({npVolumes[0].shape} != {i}: {npVolumes[i].shape}).")

        halfXDim = npVolumes[0].shape[2]//2

        npVolumesRight = [ npVolume[:, None, ..., :halfXDim] for npVolume in npVolumes ]
        npVolumesLeft = [ npVolume[:, None, ..., halfXDim:] for npVolume in npVolumes ]

        batchRight = self._resize_image(np.concatenate(tuple(npVolumesRight), axis=1)).astype(np.float32)
        batchLeft = self._resize_image(np.concatenate(tuple(npVolumesLeft), axis=1)).astype(np.float32)

        batchLeft = torch.from_numpy(batchLeft).to(self.GetDevice())
        batchRight = torch.from_numpy(batchRight).to(self.GetDevice())

        #softmax = nn.Softmax(dim=1).to(self.GetDevice())
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            self.net.eval()

            probLeft = softmax(self.net(batchLeft))
            probRight = softmax(self.net(batchRight))

            self.net.train()

            #print(probLeft.shape)
            #print(probRight.shape)

        probRight = self._pad_image(probRight.cpu().numpy(), [npVolumesRight[0].shape[0], probRight.shape[1]] + list(npVolumesRight[0].shape[2:]))
        probLeft = self._pad_image(probLeft.cpu().numpy(), [npVolumesLeft[0].shape[0], probLeft.shape[1]] + list(npVolumesLeft[0].shape[2:]))

        prob = np.concatenate((probRight, probLeft), axis=-1)
        prob = prob.transpose(0,2,3,1)

        labelMap = prob.argmax(axis=3).astype(np.int16)

        sitkProb = sitk.GetImageFromArray(prob)
        sitkLabelMap = sitk.GetImageFromArray(labelMap)

        sitkProb.SetSpacing(sitkVolumes[0].GetSpacing())
        sitkProb.SetDirection(sitkVolumes[0].GetDirection())
        sitkProb.SetOrigin(sitkVolumes[0].GetOrigin())

        sitkLabelMap.SetSpacing(sitkVolumes[0].GetSpacing())
        sitkLabelMap.SetDirection(sitkVolumes[0].GetDirection())
        sitkLabelMap.SetOrigin(sitkVolumes[0].GetOrigin())

        return sitkProb, sitkLabelMap

    def Test(self,valList):
        if isinstance(valList, str):
            with open(valList, "rt", newline='') as f:
                patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]
        else:
            patientIds = valList

        #patientIds = patientIds[:5]

        metricsByPatient = dict()

        i = 0
        for patientId in patientIds:
            print(f"Info: Running '{patientId}' ...", flush=True)

            maskFile = os.path.join(self.GetDataRoot(), "Masks", patientId, "mask_aligned.nii.gz")

            # First DICE!
            gtMask = LoadMask(maskFile, self.numClasses, dilateUnknown=False)
            npGtMask = sitk.GetArrayFromImage(gtMask)

            probMap, labelMap = self.RunOne(patientId)

            #npProbMap = sitk.GetArrayFromImage(probMap)
            #npProbMap = npProbMap[:,:,:,-1]

            if self.numClasses == 4:
                labelMap = CleanUpMask(labelMap)
                #npLabelMap = sitk.GetArrayViewFromImage(labelMap) # XXX: Very slow without making it numpy
                #npProbMap[npLabelMap == 0] = 0

            metricsByPatient[patientId] = RCCMetrics.evaluate(gtMask, labelMap, probMap)
            print("")
            print(metricsByPatient[patientId])
            print("")

        return metricsByPatient

    def Train(self,trainList,valPerc=0.0,snapshotRoot="snapshots", startEpoch=0, seed=6112, use_monai=False):
        batchSize=16
        # Info: label counts = [30077046   319095    50418    80929]
        # Info: label counts = [150327549   1658189    308158    409592]
        #labelWeights = torch.Tensor([ 0.25/30077046, 0.25/319095, 0.25/50418, 0.25/80929 ])*50418
        #labelWeights = torch.Tensor([ 0.25/150327549, 0.25/1658189, 0.25/308158, 0.25/409592 ])*308158
        #labelWeights = torch.Tensor([0.25/131337525,   0.25/1457293,    0.25/267255,    0.25/384071])*384071.0
        labelWeights = torch.Tensor([1.0]*self.numClasses).type(torch.float32)
        #labelWeights = torch.Tensor([0.00363469, 0.28188787, 1.,         0.94270076]).type(torch.float32)
        ShowWarnings(False)
        #labelWeights = torch.Tensor(ComputeLabelWeights(self.GetDataRoot(), trainList, numClasses=self.numClasses))
        #numEpochs=300
        numEpochs=1000
        #numEpochs=51

        if startEpoch < 0:
            startEpoch=self.epoch

        print(f"Info: batchSize = {batchSize}")
        print(f"Info: numClasses = {self.numClasses}")
        #print(f"Info: saveSteps = {self.saveSteps}")
        print(f"Info: valSteps = {self.valSteps}")
        print(f"Info: labelWeights = {labelWeights}")
        print(f"Info: dilateUnknown = {self.dilateUnknown}")

        # Count parameters
        numParams = sum((params.numel() for params in self.net.parameters()))
        numTrainable = sum((params.numel() for params in self.net.parameters() if params.requires_grad))
        print(f"There are {numParams} parameters total in this model ({numTrainable} are trainable)")

        with open(trainList, mode="rt", newline="") as f:
            patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]

        valList = None
        if valPerc > 0.0:
            mid = max(1, int(valPerc*len(patientIds)))
            valList = patientIds[:mid]
            trainList = patientIds[mid:]
        else:
            trainList = patientIds

        if use_monai:
            print("Info: Using MONAI ImageBatcher", flush=True)
            from ImageBatcher_monai import ImageBatcher
        else:
            from ImageBatcher import ImageBatcher

        imageBatcher = ImageBatcher(self.GetDataRoot(), trainList, batchSize, numClasses=self.numClasses, dilateUnknown=self.dilateUnknown, seed=seed)

        criterion = nn.CrossEntropyLoss(ignore_index=-1,weight = labelWeights).to(self.GetDevice())
        #criterion2 = DiceLoss(ignore_index=0, p=1, smooth=1e-3).to(self.GetDevice())

        learningRate = 1e-3

        #criterion = ops.sigmoid_focal_loss
        optimizer = optim.Adam(self.net.parameters(), lr = learningRate)

        bestDsc = None

        #trainLosses = np.ones([numEpochs])*1000.0
        #valAUCs = np.zeros([numEpochs])

        if not os.path.exists(snapshotRoot):
            os.makedirs(snapshotRoot)

        try:
            imageBatcher.start()
            for e in range(startEpoch,numEpochs):
            #for e in range(148, numEpochs):

                self.epoch = e

                runningLoss = 0.0
                count = 0

                for xbatch, ybatch in imageBatcher:
                    #break
                    #print(xbatch.shape)

                    xbatch = xbatch.to(self.GetDevice())
                    ybatch = ybatch.to(self.GetDevice())

                    optimizer.zero_grad()            

                    outputs = self.net(xbatch)

                    with NotDeterministic():
                        loss = criterion(outputs, ybatch)

                        #knownMask = (ybatch >= 0)
                        #yohbatch = make_one_hot(ybatch[knownMask].view(1,1,-1), num_classes=self.numClasses).to(self.GetDevice())

                        #outputsoh = outputs[knownMask[:,None,:,:].repeat(1,self.numClasses,1,1)].view(1,self.numClasses, -1).to(self.GetDevice())
                        #loss += criterion2(outputsoh, yohbatch)

                        loss.backward()

                    optimizer.step()

                    runningLoss += loss
                    count += 1

                    #print(f"loss = {loss.item()}", flush=True)

                if count > 0:
                    runningLoss /= count

                #snapshotFile=os.path.join(snapshotRoot, f"epoch_{e}.pt")
                snapshotFile=os.path.join(snapshotRoot, "model.pt")
                #rocFile=os.path.join(snapshotRoot, f"validation_roc_{e}.txt")
                rocFile=os.path.join(snapshotRoot, "validation_roc.txt")
                #diceFile=os.path.join(snapshotRoot, f"dice_stats_{e}.txt")
                diceFile=os.path.join(snapshotRoot, "dice_stats.txt")

                # This is the original behavior... this eats up too much disk space though
                #if ((e+1) % self.saveSteps) == 0:
                #    print(f"Info: Saving {snapshotFile} ...", flush=True)
                #    self.SaveModel(snapshotFile)
                #else:
                #    print(f"Info: Skipping saving {snapshotFile}.", flush=True)

                # For debugging
                #self.LoadModel(snapshotFile)

                #trainLosses[e] = runningLoss

                print(f"Info: Epoch = {e}, training loss = {runningLoss}", flush=True)

                if self.valSteps > 0 and ((e+1) % self.valSteps) == 0: 
                    metricsByPatient = self.Test(valList)

                    diceStats = extract_metrics(metricsByPatient, 'dsc')
                    msdStats = extract_metrics(metricsByPatient, 'assd')
                    hdStats = extract_metrics(metricsByPatient, 'hd')
                    totalCounts, missedCounts, extraCounts = extract_counts(metricsByPatient)
                    roc = calculate_roc(metricsByPatient)

                    #valAUCs[e] = roc[3]

                    if bestDsc is None or diceStats[0][3] > bestDsc:
                        print(f"Info: Got better tumor DSC: {diceStats[0][3]} > {bestDsc}")
                        print(f"Info: Saving '{snapshotFile}' ...")

                        self.SaveModel(snapshotFile)

                        bestDsc = diceStats[0][3]

                        with open(rocFile, mode="wt", newline="") as f:
                            f.write("# Threshold\tFPR\tTPR\n")

                            for threshold, fpr, tpr in zip(roc[0], roc[1], roc[2]):
                                f.write(f"{threshold}\t{fpr}\t{tpr}\n")

                            f.write(f"# AUC = {roc[3]}\n")

                        with open(diceFile, mode="wt", newline="") as f:
                            #for patientId in allDices:
                            #    f.write(f"{patientId}: {allDices[patientId]}\n")
                            for patientId, metricsByLabel in metricsByPatient.items():
                                f.write(f"{patientId}: {metricsByLabel[3]['dsc'] if 'dsc' in metricsByLabel[3] else -1.0}\n")

                            f.write(f"\nDice stats: {diceStats[0]} +/- {diceStats[1]}\n")

                    print(f"Info: Epoch = {e}, training loss = {runningLoss}, learning rate = {learningRate}, validation AUC = {-1}, validation dices = {diceStats[0]} +/- {diceStats[1]}, validation msd = {msdStats[0]} +/- {msdStats[1]}, validation hd = {hdStats[0]} +/- {hdStats[1]}, tumor counts = {totalCounts[3]} / {missedCounts[3]}, auc = {roc[3]}", flush=True)
        except:
            imageBatcher.stop()
            raise sys.exc_info()[1]

        imageBatcher.stop()
 
        #return trainLosses, valAUCs

