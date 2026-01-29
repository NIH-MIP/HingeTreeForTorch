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
import argparse
import hashlib
import random
import numpy as np
import torch
import glob
from RCCSeg import RCCSeg
from Deterministic import set_deterministic

def seed(seedStr):
    theSeed = int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)
    random.seed(theSeed)
    np.random.seed(theSeed) # Bad way to do this!
    #torch.random.manual_seed(seed)
    #torch.cuda.random.manual_seed(seed)
    torch.manual_seed(theSeed)
    return theSeed

def find_last_snapshot(snapshotDir):
    startSnapshotFile=os.path.join(snapshotDir, "model.pt")

    # NOTE: Epoch stored in new format parameters file
    if os.path.exists(startSnapshotFile):
        return startSnapshotFile, -1

    startEpoch=0
    startSnapshotFile=None
    for snapshotFile in glob.glob(os.path.join(snapshotDir, "epoch_*.pt")):
        base=os.path.splitext(os.path.basename(snapshotFile))[0]
        epoch=int(base.split("_")[-1])+1

        if epoch > startEpoch:
            startEpoch = epoch
            startSnapshotFile = snapshotFile

    return startSnapshotFile, startEpoch

def main(dataRoot, trainList, snapshotDir, seedStr="rcc0", continueTraining=False, use_expand=False, use_monai=False):
    #dataRoot="/data/AIR/RCC/NiftiNew"
    #dataRoot="/data/AIR/RCC/Nifti"
    #testRoot="/data/AIR/layns/NIHAnonMeshes"
    #trainList=os.path.join(dataRoot, "trainList.txt")
    #valList=os.path.join(dataRoot, "valList.txt")

    set_deterministic(True)
    theSeed=seed(seedStr)

    print(f"trainList = {trainList}", flush=True)
    print(f"dataRoot = {dataRoot}", flush=True)
    print(f"seed = '{seedStr}'", flush=True)
    print(f"snapshotDir = {snapshotDir}", flush=True)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    cad = RCCSeg(numClasses=4, use_expand=use_expand)

    cad.SetDevice("cuda:0")

    cad.SetDataRoot(dataRoot)
    #cad.SetTestRoot(testRoot)
    #cad.LoadLesionDemographics(demographicsCsv)

    #cad.Train(trainList, valList, snapshotRoot="snapshots_FourChannels_Binary_Weighted_RelabeledUnknowns")
    #modelPath=os.path.join(snapshotDir, "epoch_128.pt")

    startEpoch=0

    if continueTraining:
        continueSnapshot, startEpoch = find_last_snapshot(snapshotDir)

        if continueSnapshot is not None:
            print(f"Info: Continuing from '{continueSnapshot}' (start epoch = {startEpoch}) ...")
            cad.LoadModel(continueSnapshot)
        else:
            print(f"Info: No snapshots found. Starting from scratch ...")

    cad.Train(trainList, valPerc=0.1, snapshotRoot=snapshotDir, startEpoch=startEpoch, seed=theSeed, use_monai=use_monai)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCC Project")
    parser.add_argument("--data-root", dest="dataRoot", required=True, type=str, help="Data root.")
    parser.add_argument("--train-list", dest="trainList", required=True, type=str, help="Training list file.")
    parser.add_argument("--snapshot-dir", dest="snapshotDir", required=True, type=str, help="Snapshot directory.")
    parser.add_argument("--seed", dest="seedStr", default="rcc", type=str, help="Seed string.")
    parser.add_argument("--continue", dest="continueTraining", default=False, action="store_true", help="Continue from last snapshot.")
    parser.add_argument("--use-expand", dest="use_expand", default=False, action="store_true", help="Use HingeTree.expand instead of fold.")
    parser.add_argument("--use-monai", dest="use_monai", default=False, action="store_true", help="Use MONAI training data loader.")

    args = parser.parse_args()

    main(**vars(args))

