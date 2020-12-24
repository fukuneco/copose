from config_reader import config_reader
import numpy as np
import keras
import cv2

from Mod.OPModel_v201130 import *      # openpose用のモデルを読み込む
from Mod.MyMod_v201130 import *
from Mod.MyLossFunc_v201202 import *   # 類似度の評価関数
from Mod.OPCalc_v201130 import *

############################################### ここで評価法を選択
#ext_col = "default"       # 
#ext_col = "distance"      # ユークリッド距離
ext_col = "similarity"    # コサイン類似度
###############################################

RANK_SAVE_DIR = "./002_ranking/"                       # ランキング結果の保存先

#######################################################################
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

########################################################################
# @oriImg: 画像データ
def getJoints(model, oriImg):
    your_joints = np.zeros((18, 2))              # (x, y)の18関節
    best_human_idx = 0                 # 本来は読み込み画像数のリストだったが，1データのみなので変更 = np.zeros(len(loaded_filename))
    best_human_pos = {}
    non_joint_fn = [] # 関節が見つからなかったファイル名

    # Load Configuration
    param, model_params = config_reader()
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    # heatmap計算
    heatmap_avg, paf_avg = getHeatMap_Avg(model, oriImg, model_params, multiplier, log=False)

    # all_peaksを計算するのにheatmap_avgが必要
    all_peaks, peak_counter = getAllPeaks(heatmap_avg, param)
    connection_all, special_k = getConnectionAll(paf_avg, all_peaks, oriImg, param, mapIdx, limbSeq)

    # subset: 各人の関節id(順に18個，未検出の場合は-1が入る), 確度, 検出された関節総数(最大18)
    # candidate: 関節座標x, y, ?, id　
    subset, candidate = getSubset(all_peaks, mapIdx, special_k, connection_all, limbSeq)

    print('subset:')
    print(subset)

    if len(subset) == 0: # 空判定
        print('Cannot get subset...')
        non_joint_fn.append(img_fn)
        print("------------------------------------------")
        return -1


    # 最も大きい人物を抽出. target_joint:1(首)からの距離で探す
    best_human_idx = getBestHumanId(subset, candidate, target_joint =1, log = True)
    #print(best_human_idx)


    # 関節のxy座標を取得
    img_fn = "test"
    best_human_pos = getHumanJoint(best_human_idx, subset, candidate, img_fn)
    print(best_human_pos)

    # 比較用の変数に格納
    for i in range(18):
        dat = best_human_pos.get(i)
        if dat is None:
            your_joints[i][0] = None # 存在しない場合はマイナス座標値
            your_joints[i][1] = None # 存在しない場合はマイナス座標値
        else:
            your_joints[i][0] = dat[0].astype(int)
            your_joints[i][1] = dat[1].astype(int)

    return your_joints
###################################################################
# @loaded_joints: pickleから読み込んだファイル名と関節座標の辞書データ
# @your_joints:ファイル選択から読み込んだ関節座標のリスト

def getSimFileList(loaded_joints, your_joints):
    import math
    import pandas as pd

    loaded_best_human_pos= np.zeros((18,2)) # x,y 18関節
    flg_first_NF_your_joints = False        # 入力ファイルの関節不足情報フラグ
    ORG_ID = 1 # 1:首 14:右目

    # filename  : pickleから読み込んだファイル名
    # default   : 上のリンク参考、全ての点の絶対座標の距離
    # distance  : 首の座標を揃えた場合の全ての点の距離
    # similarity: 関節間のベクトルを用いたコサイン類似度
    df = pd.DataFrame(columns=['filename', 'default', 'distance', 'similarity'])

    for i, key in enumerate (loaded_joints.keys()): # pickleに含まれる画像の数ループ
        print('---')
        d = 0
        d2= 0
        nj_id = [] #関節が見つからないidを補完する
        
        #基準座標を先に取得, 
        neck_tmp = loaded_joints[key].get(ORG_ID) # 1: 首のid
        
        if neck_tmp is None:
            print("Neck nothing...")
            neck_x = -1
            neck_y = -1
            if ext_col != "default":
                continue            # 首がなかったら処理しなくてよいのでは，
        else:
            neck_x = neck_tmp[0]
            neck_y = neck_tmp[1]
        
        for j in range(18): # 関節数ループ
            
            dat = loaded_joints[key].get(j)
            
            # 事前評価データにNoneが含まれる場合の処理
            if dat is None:
                nj_id.append(j)
                continue
            
            loaded_best_human_pos[j][0] = dat[0] # x座標
            loaded_best_human_pos[j][1] = dat[1] # y座標  

            ev = np.array([your_joints[j][0], your_joints[j][1]])
            if np.isnan(ev).any():
                if flg_first_NF_your_joints == False: #最初だけ入力ファイルの関節不足情報を表示
                    print("Joint Not Found: %s, %d" %("your file", j))
                    flg_first_NF_your_joints = True
                continue
            
            # 類似度評価関数 ##############################################
            # inputと比較両方に存在するポイントの距離を計算
            d += (abs(your_joints[j][0] - loaded_best_human_pos[j][0]) \
                + abs(your_joints[j][1] - loaded_best_human_pos[j][1]))
            
            # 首の座標でそろえて、各々の距離を計算。
            if neck_x >= 0:
                your_x = your_joints[j][0] - your_joints[ORG_ID][0]
                your_y = your_joints[j][1] - your_joints[ORG_ID][1]
                load_x = loaded_best_human_pos[j][0] - neck_x
                load_y = loaded_best_human_pos[j][1] - neck_y
            
            d2 += math.sqrt((your_x - load_x)**2+(your_y - load_y)**2)
            ##############################################################
        
        if len(nj_id) > 1:
            print("Joint Not Found: %s" %key)
            #print(nj_id)
            
        #similarity = getCosSimilarity(loaded_best_human_pos, tmp)
        similarity, similarity_inv = getCosSimilarity(loaded_best_human_pos, your_joints, neck_x)
        
        # ファイル名と評価関数を保存
        df.loc[i] = [key, d, d2, similarity]

    print("results...")

    # 並び替え
    mkdir(RANK_SAVE_DIR)

    if ext_col == "distance":
        df_s = df.sort_values('distance') # 昇順
    elif ext_col == "similarity":
        df_s = df.sort_values('similarity', ascending = False) # 降順
    else:
        df_s = df.sort_values('default') # 昇順

    print(df_s)
    df_s.to_csv(RANK_SAVE_DIR + "000_ranking.csv",columns=['filename', ext_col])

    df_s=df_s.reset_index(drop=True)  # インデックスを振り直し

    return df_s