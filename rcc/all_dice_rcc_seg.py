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
import glob
from RCCSeg import RCCSeg, extract_metrics, extract_counts, calculate_roc
from rcc_common import LoadImage, SaveImage, LoadMask, CleanUpMask, ShowWarnings
from roc import AverageROC, ComputeROC
import torch
import SimpleITK as sitk
import numpy as np
from Deterministic import set_deterministic

def extract_all_metrics(metricsBySplit, key, defaultVal=-1):
    mean = [defaultVal]*4
    std = [defaultVal]*4
    median = [defaultVal]*4

    for label in (1,2,3):

        valuesForLabel = []

        for metricsByPatient in metricsBySplit:
            if not any(key in metricsByLabel[label] for metricsByLabel in metricsByPatient.values()):
                continue

            if key == "mask_dsc":
                # mask_dsc is just a single value
                valuesForLabel = np.concatenate([valuesForLabel] + [ [ metricsByLabel[label][key] ] for metricsByLabel in metricsByPatient.values() if key in metricsByLabel[label] ])
            else:
                valuesForLabel = np.concatenate([valuesForLabel] + [ metricsByLabel[label][key] for metricsByLabel in metricsByPatient.values() if key in metricsByLabel[label] ])

        mean[label] = valuesForLabel.mean()
        std[label] = valuesForLabel.std()
        median[label] = np.median(valuesForLabel)

    return mean, std, median

def extract_all_counts(metricsBySplit):
    totalCounts = np.zeros([4], dtype=np.int32)
    missedCounts = np.zeros([4], dtype=np.int32)
    extraCounts = np.zeros([4], dtype=np.int32)

    for metricsByPatient in metricsBySplit:
        totalCountsForSplit, missedCountsForSplit, extraCountsForSplit = extract_counts(metricsByPatient)
        totalCounts += totalCountsForSplit
        missedCounts += missedCountsForSplit
        extraCounts += extraCountsForSplit

    return totalCounts, missedCounts, extraCounts

def calculate_roc_stats(metricsBySplit):
    rocs = []

    for metricsByPatient in metricsBySplit:
        scoreAndLabels = np.concatenate([ metricsByLabel[3]["tumor_detections"] for metricsByLabel in metricsByPatient.values() if "tumor_detections" in metricsByLabel[3] and len(metricsByLabel[3]["tumor_detections"]) > 0 ], axis=0)
        scores = torch.from_numpy(scoreAndLabels[:, 0])
        labels = torch.from_numpy(scoreAndLabels[:, 1])

        roc = ComputeROC(scores, labels)

        rocs.append(roc)

    allFprs, meanTprs, stdTprs, auc, aucStd = AverageROC(rocs)

    return allFprs, meanTprs, stdTprs, auc, aucStd

def main():
    ShowWarnings(False)
    #dataRoot="/data/AIR/RCC/Nifti.bak6"
    #dataRoot="/data/AIR/RCC/NiftiNew"
    #dataRoot="/data/AIR/RCC/NiftiNew"
    dataRoot="/scratch/cluster_scratch/layns/RCC/NiftiNew"
    #dataRoot="/data/AIR/RCC/Temp2/Nifti"
    #testList=os.path.join(dataRoot, "valList.txt")
    #testList=os.path.join(dataRoot, "trainList.txt")
    #testList=os.path.join(dataRoot, "good.txt")

    set_deterministic(True)

    cad = RCCSeg(numClasses=4, use_expand=False)
    cad.SetDevice("cuda:0")
    cad.SetDataRoot(dataRoot)

    numSplits = 10
    metricsBySplit = [None]*numSplits

    meanMaskDiceBySplit = np.full([numSplits, cad.numClasses], -1.0)
    meanDiceBySplit = np.full([numSplits, cad.numClasses], -1.0)
    meanAssdBySplit = np.full([numSplits, cad.numClasses], -1.0)
    meanHdBySplit = np.full([numSplits, cad.numClasses], -1.0)

    for f in range(numSplits):
        #modelFolder=f"/data/AIR/RCC/ISBI/snapshots_fold1_randomSplit{f+1}"
        #modelFolder=f"/data/AIR/RCC/ISBI/snapshots_weighted_easyhard_deterministic_hingeforest_randomSplit{f+1}"
        #modelFolder=f"/data/AIR/RCC/ISBI/snapshots_weighted_easyhard_deterministic_hingeforest_vggblock3_randomSplit{f+1}"
        #modelFolder=f"/data/AIR/RCC/ISBI/snapshots_unweighted_easyhard_deterministic_hingeforest_depth7_vggblock3_randomSplit{f+1}"
        #modelFolder=f"/scratch/cluster_scratch/layns/RCC/Models/rccseg_unweighted_easyhard_deterministic_depth7_hingeforest_vggblock3_randomSplit{f+1}"
        #modelFolder=f"/scratch/cluster_scratch/layns/RCC/Models/rccseg_unweighted_easyhard_deterministic_depth7_hingeforest_vggblock3_fold_randomSplit{f+1}"
        modelFolder=f"Models/snapshots_unweighted_easyhard_deterministic_hingeforest_depth7_vggblock3_randomSplit{f+1}"
        bestFile=os.path.join(modelFolder, "bestDice.txt")        
        testList=os.path.join(dataRoot, f"test_isbi2022_easyhard_randomSplit{f+1}.txt")

        bestEpoch=-1
        if os.path.exists(bestFile):
            with open(bestFile, mode="rt", newline="") as g:
                line = next(iter(g))
                bestEpoch = int(line.split(" ")[-1])

            #bestEpoch=299
            bestModel=os.path.join(modelFolder, f"epoch_{bestEpoch}.pt")
        else:
            # XXX: The bestDice really should exist for this... but the model was stored on github with just one snapshot
            try:
                bestModel=next(iter(glob.glob(os.path.join(modelFolder, "epoch_*.pt"))))
            except StopIteration:
                bestModel=os.path.join(modelFolder, f"model.pt")

        print(f"Info: Loading '{bestModel}' ...")
        cad.LoadModel(bestModel)

        if bestEpoch < 0:
            bestEpoch = cad.epoch
            print(f"Info: epoch = {bestEpoch}")

        metricsByPatient = cad.Test(testList)
        metricsBySplit[f] = metricsByPatient

        diceStats = extract_metrics(metricsByPatient, 'dsc')
        maskDiceStats = extract_metrics(metricsByPatient, 'mask_dsc')
        assdStats = extract_metrics(metricsByPatient, 'assd')
        hdStats = extract_metrics(metricsByPatient, 'hd')
        rocStats = calculate_roc(metricsByPatient)
        counts = extract_counts(metricsByPatient)

        meanDiceBySplit[f, :] = diceStats[0]
        meanMaskDiceBySplit[f, :] = maskDiceStats[0]
        meanAssdBySplit[f, :] = assdStats[0]
        meanHdBySplit[f, :] = hdStats[0]

        print(f"Dice scores: {diceStats}")
        print(f"Mask dice scores: {maskDiceStats}")
        print(f"ASSD scores: {assdStats}")
        print(f"HD scores: {hdStats}")
        print(f"Total: {counts[0]}")
        print(f"Missed: {counts[1]}")
        print(f"Extra: {counts[2]}")
        print(f"AUC: {rocStats[3]}")

    diceStats = extract_all_metrics(metricsBySplit, 'dsc')
    maskDiceStats = extract_all_metrics(metricsBySplit, 'mask_dsc')
    assdStats = extract_all_metrics(metricsBySplit, 'assd')
    hdStats = extract_all_metrics(metricsBySplit, 'hd')
    rocStats = calculate_roc_stats(metricsBySplit)
    counts = extract_all_counts(metricsBySplit)

    #print(f"Overall ROC:\nfpr = {meanROC[0]}\ntpr = {meanROC[1]}\nstd = {meanROC[2]}\nauc = {meanROC[3]}")
    print(f"Overall structure performance: {diceStats[0]} +/- {diceStats[1]}, median = {diceStats[2]}")
    print(f"Overall structure mask performance: {maskDiceStats[0]} +/- {maskDiceStats[1]}, median = {maskDiceStats[2]}")
    print(f"Overall structure ASSD performance: {assdStats[0]} +/- {assdStats[1]}, median = {assdStats[2]}")
    print(f"Overall structure HD performance: {hdStats[0]} +/- {hdStats[1]}, median = {hdStats[2]}")
    print(f"Overall structure total: {counts[0]}")
    print(f"Overall structure missed: {counts[1]}")
    print(f"Overall structure extra: {counts[2]}")
    print(f"Overall ROC performance: {rocStats[3]} +/- {rocStats[4]}")

    print(f"Overall mean performance: {meanDiceBySplit.mean(axis=0)} +/- {meanDiceBySplit.std(axis=0)}")
    print(f"Overall mean mask performance: {meanMaskDiceBySplit.mean(axis=0)} +/- {meanMaskDiceBySplit.std(axis=0)}")
    print(f"Overall mean ASSD performance: {meanAssdBySplit.mean(axis=0)} +/- {meanAssdBySplit.std(axis=0)}")
    print(f"Overall mean HD performance: {meanHdBySplit.mean(axis=0)} +/- {meanHdBySplit.std(axis=0)}")

    rocFile="mean_roc.txt"
    #rocFile="mean_roc_weighted_depth7_hingeforest_vggblock3.txt"
    print(f"\nWriting {rocFile} ...")
    with open(rocFile, mode="wt") as f:
        f.write("# FPR TPR TPRSTD\n")
        for fpr, tpr, tprStd in zip(rocStats[0], rocStats[1], rocStats[2]):
            f.write(f"{fpr} {tpr} {tprStd}\n")

        f.write(f"# AUC = {rocStats[3]} +/- {rocStats[4]}\n")

    filePairs = [("dice_raw.txt", meanDiceBySplit),
        ("mask_dice_raw.txt", meanMaskDiceBySplit),
        ("assd_raw.txt", meanAssdBySplit),
        ("hd_raw.txt", meanHdBySplit)]

    for filePath, meanPerfBySplit in filePairs:
        print(f"\nWriting {filePath} ...")

        with open(filePath, mode="wt") as g:
            if meanPerfBySplit.shape[1] == 4:
                g.write("Background,kidney,cyst,tumor\n")

            for f in range(meanPerfBySplit.shape[0]):
                g.write(f"{meanPerfBySplit[f,0]}")
                for c in range(1,meanPerfBySplit.shape[1]):
                    g.write(f",{meanPerfBySplit[f,c]}")
                g.write("\n")

if __name__ == "__main__":
    main()

